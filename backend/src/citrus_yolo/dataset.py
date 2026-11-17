"""Tudo que mexe no dataset: rótulos YOLO, checagens, divisão treino/val e data.yaml.

Fluxo do `citrus-yolo preparar`:
  dataset/bruto/{images,labels}  →  valida cada rótulo  →  remove duplicatas (hash)
  →  divide 80/20 estratificado  →  dataset/{images,labels}/{train,val} + data.yaml
  →  confere se as inéditas não são cópia de nada  →  resultados/dataset/resumo_dataset.json
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Config, ErroProjeto

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MARCADOR_SINTETICO = "SINTETICO.txt"
PASTAS_ROTULO_INEDITAS = ("rotulos", "labels")


@dataclass(frozen=True)
class Caixa:
    """Uma linha do .txt: classe x_centro y_centro largura altura (tudo normalizado 0..1)."""

    classe: int
    xc: float
    yc: float
    w: float
    h: float

    def xyxy(self, largura: int, altura: int) -> tuple[float, float, float, float]:
        return (
            (self.xc - self.w / 2) * largura,
            (self.yc - self.h / 2) * altura,
            (self.xc + self.w / 2) * largura,
            (self.yc + self.h / 2) * altura,
        )

    def linha(self) -> str:
        return f"{self.classe} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class Amostra:
    imagem: Path
    rotulo: Path
    caixas: list[Caixa]
    hash: str

    @property
    def classes(self) -> set[int]:
        return {c.classe for c in self.caixas}


# --------------------------------------------------------------------------- rótulos


def ler_rotulo(caminho: Path, n_classes: int) -> list[Caixa]:
    caixas = []
    texto = Path(caminho).read_text(encoding="utf-8-sig")
    for numero, linha in enumerate(texto.splitlines(), start=1):
        linha = linha.strip()
        if not linha:
            continue
        partes = linha.split()
        onde = f"{Path(caminho).name}, linha {numero}"
        if len(partes) != 5:
            if len(partes) > 5 and (len(partes) - 1) % 2 == 0:
                raise ErroProjeto(
                    f"{onde}: isso parece um polígono (segmentação), não uma caixa. Exporte no formato "
                    "'YOLOv8' de detecção: classe x_centro y_centro largura altura."
                )
            raise ErroProjeto(
                f"{onde}: esperava 5 valores (classe x_centro y_centro largura altura), achei {len(partes)}."
            )
        try:
            numeros = [float(p) for p in partes]
        except ValueError as exc:
            raise ErroProjeto(f"{onde}: tem valor que não é número: '{linha}'.") from exc
        if not numeros[0].is_integer():
            raise ErroProjeto(f"{onde}: a classe precisa ser um número inteiro, achei {partes[0]}.")
        classe = int(numeros[0])
        if not 0 <= classe < n_classes:
            raise ErroProjeto(
                f"{onde}: classe {classe} fora do intervalo 0..{n_classes - 1}. "
                "Confira se a ordem das classes em configs/projeto.yaml bate com a da ferramenta de anotação."
            )
        xc, yc, w, h = numeros[1:]
        if any(v < 0 or v > 1 for v in (xc, yc, w, h)):
            raise ErroProjeto(
                f"{onde}: coordenadas precisam estar normalizadas entre 0 e 1 (achei {xc}, {yc}, {w}, {h}). "
                "Se estão em pixels, a exportação não foi no formato YOLO."
            )
        if w <= 0 or h <= 0:
            raise ErroProjeto(f"{onde}: largura e altura da caixa precisam ser maiores que zero.")
        caixas.append(Caixa(classe, xc, yc, w, h))
    return caixas


def escrever_rotulo(caminho: Path, caixas: list[Caixa]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("".join(c.linha() + "\n" for c in caixas), encoding="utf-8")


def listar_imagens(pasta: Path) -> list[Path]:
    if not pasta.is_dir():
        return []
    return sorted(p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES_IMAGEM)


def rotulo_da_inedita(imagem: Path) -> Path | None:
    """Gabarito opcional de uma imagem inédita (ineditas/rotulos/<nome>.txt)."""
    for pasta in PASTAS_ROTULO_INEDITAS:
        candidato = imagem.parent / pasta / f"{imagem.stem}.txt"
        if candidato.is_file():
            return candidato
    return None


def sha1(caminho: Path) -> str:
    h = hashlib.sha1()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def origem(nome_arquivo: str) -> str:
    """Agrupa cópias aumentadas do Roboflow ("foto_jpg.rf.<hash>.jpg") pela foto original.

    Sem isso, duas versões da mesma foto podem cair uma no treino e outra na validação, e a
    métrica de validação fica otimista demais.
    """
    return Path(nome_arquivo).stem.split(".rf.")[0]


# --------------------------------------------------------------------------- leitura do bruto


def carregar_bruto(cfg: Config) -> tuple[list[Amostra], list[str]]:
    pasta = cfg.p("bruto")
    pasta_img, pasta_rot = pasta / "images", pasta / "labels"
    imagens = listar_imagens(pasta_img)
    if not imagens:
        raise ErroProjeto(
            f"Nenhuma imagem em {cfg.relativo(pasta_img)}. Coloque as fotos lá e os .txt em "
            f"{cfg.relativo(pasta_rot)} (ou rode: citrus-yolo importar <pasta_exportada>)."
        )

    avisos: list[str] = []
    amostras: list[Amostra] = []
    sem_rotulo: list[str] = []
    vistos: dict[str, str] = {}
    for imagem in imagens:
        rotulo = pasta_rot / f"{imagem.stem}.txt"
        if not rotulo.is_file():
            sem_rotulo.append(imagem.name)
            continue
        caixas = ler_rotulo(rotulo, len(cfg.classes))
        assinatura = sha1(imagem)
        if assinatura in vistos:
            avisos.append(f"{imagem.name} é cópia idêntica de {vistos[assinatura]}; ignorei para não vazar.")
            continue
        vistos[assinatura] = imagem.name
        if not caixas:
            avisos.append(f"{imagem.name} não tem nenhuma caixa (imagem de fundo); não conta como anotada.")
        amostras.append(Amostra(imagem, rotulo, caixas, assinatura))

    if sem_rotulo:
        lista = ", ".join(sem_rotulo[:5]) + (" ..." if len(sem_rotulo) > 5 else "")
        raise ErroProjeto(
            f"{len(sem_rotulo)} imagem(ns) sem o .txt correspondente em {cfg.relativo(pasta_rot)}: {lista}. "
            "Anote essas fotos ou tire-as de bruto/images."
        )
    orfaos = sorted({p.stem for p in pasta_rot.glob("*.txt")} - {i.stem for i in imagens} - {"classes"})
    if orfaos:
        avisos.append(f"{len(orfaos)} rótulo(s) sem imagem foram ignorados (ex.: {orfaos[0]}.txt).")
    return amostras, avisos


# --------------------------------------------------------------------------- divisão


def dividir(amostras: list[Amostra], proporcao_val: float, semente: int) -> tuple[list[Amostra], list[Amostra]]:
    """Divisão estratificada e reprodutível.

    1. Junta as cópias da mesma foto (ver `origem`) num grupo só, que vai inteiro para um lado.
    2. Cada grupo é "etiquetado" pela classe mais rara que ele contém, para a classe minoritária
       também aparecer na validação.
    3. Dentro de cada etiqueta, embaralha com a semente e separa ~proporcao_val para validação.
    """
    grupos: dict[str, list[Amostra]] = defaultdict(list)
    for amostra in amostras:
        grupos[origem(amostra.imagem.name)].append(amostra)

    frequencia = Counter(c.classe for a in amostras for c in a.caixas)
    estratos: dict[int, list[list[Amostra]]] = defaultdict(list)
    for chave in sorted(grupos):
        classes = set().union(*(a.classes for a in grupos[chave]))
        etiqueta = min(classes, key=lambda c: (frequencia[c], c)) if classes else -1
        estratos[etiqueta].append(grupos[chave])

    rng = random.Random(semente)
    treino: list[Amostra] = []
    val: list[Amostra] = []
    for etiqueta in sorted(estratos):
        itens = estratos[etiqueta]
        rng.shuffle(itens)
        n_val = round(len(itens) * proporcao_val)
        if n_val == 0 and len(itens) >= 2:
            n_val = 1
        n_val = max(0, min(n_val, len(itens) - 1))  # sempre sobra pelo menos 1 grupo no treino
        for grupo in itens[:n_val]:
            val.extend(grupo)
        for grupo in itens[n_val:]:
            treino.extend(grupo)

    def ordem(a: Amostra) -> str:
        return a.imagem.name

    return sorted(treino, key=ordem), sorted(val, key=ordem)


# --------------------------------------------------------------------------- escrita


def escrever_split(cfg: Config, treino: list[Amostra], val: list[Amostra]) -> None:
    base = cfg.p("dataset")
    bruto = cfg.p("bruto")
    for sub in ("images", "labels"):
        alvo = base / sub
        if bruto == alvo or alvo in bruto.parents:
            raise ErroProjeto("caminhos.bruto não pode ficar dentro de dataset/images ou dataset/labels.")
        if alvo.exists():
            shutil.rmtree(alvo)  # pasta 100% gerada; recriar evita sobra de divisões antigas
    for nome, amostras in (("train", treino), ("val", val)):
        (base / "images" / nome).mkdir(parents=True, exist_ok=True)
        (base / "labels" / nome).mkdir(parents=True, exist_ok=True)
        for a in amostras:
            shutil.copy2(a.imagem, base / "images" / nome / a.imagem.name)
            escrever_rotulo(base / "labels" / nome / f"{a.imagem.stem}.txt", a.caixas)


def conteudo_data_yaml(classes: list[str], caminho_dataset: Path | None) -> str:
    dados: dict = {}
    if caminho_dataset is not None:
        dados["path"] = Path(caminho_dataset).resolve().as_posix()
    dados.update({"train": "images/train", "val": "images/val", "names": dict(enumerate(classes))})
    cabecalho = "# Gerado por `citrus-yolo preparar`. Não edite à mão: rode o comando de novo.\n"
    return cabecalho + yaml.safe_dump(dados, sort_keys=False, allow_unicode=True)


def escrever_data_yaml(cfg: Config) -> Path:
    destino = cfg.data_yaml
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(conteudo_data_yaml(cfg.classes, cfg.p("dataset")), encoding="utf-8")
    return destino


# --------------------------------------------------------------------------- checagens


def checar_ineditas(cfg: Config, hashes_conhecidos: dict[str, str]) -> list[Path]:
    imagens = listar_imagens(cfg.p("ineditas"))
    vazadas = []
    for img in imagens:
        assinatura = sha1(img)
        if assinatura in hashes_conhecidos:
            vazadas.append(f"{img.name} (= {hashes_conhecidos[assinatura]})")
    if vazadas:
        raise ErroProjeto(
            "Estas imagens 'inéditas' são cópias de fotos do treino/validação: "
            + ", ".join(vazadas)
            + ". Troque por fotos novas; do contrário o teste não prova nada."
        )
    return imagens


def hashes_do_split(cfg: Config) -> dict[str, str]:
    """Hash de tudo que está em dataset/images (treino + val), para checar as inéditas depois."""
    resultado = {}
    for sub in ("train", "val"):
        for img in listar_imagens(cfg.p("dataset") / "images" / sub):
            resultado[sha1(img)] = f"{sub}/{img.name}"
    return resultado


def estatisticas(amostras: list[Amostra], classes: list[str]) -> dict:
    instancias = Counter(c.classe for a in amostras for c in a.caixas)
    imagens_com = Counter(k for a in amostras for k in a.classes)
    return {
        "imagens": len(amostras),
        "imagens_anotadas": sum(1 for a in amostras if a.caixas),
        "instancias": {nome: instancias.get(i, 0) for i, nome in enumerate(classes)},
        "imagens_por_classe": {nome: imagens_com.get(i, 0) for i, nome in enumerate(classes)},
    }


def avisos_balanceamento(instancias: dict[str, int]) -> list[str]:
    avisos = []
    vazias = [nome for nome, n in instancias.items() if n == 0]
    if vazias:
        avisos.append(f"Classe(s) sem nenhuma caixa anotada: {', '.join(vazias)}. O modelo não vai aprendê-la(s).")
    com_dados = {k: v for k, v in instancias.items() if v > 0}
    if len(com_dados) >= 2:
        maior = max(com_dados, key=com_dados.get)
        menor = min(com_dados, key=com_dados.get)
        if com_dados[menor] / com_dados[maior] < 0.5:
            avisos.append(
                f"Dataset desbalanceado: {menor} tem {com_dados[menor]} caixas contra {com_dados[maior]} de "
                f"{maior}. Tente aproximar as quantidades; a classe menor tende a ter recall pior."
            )
    return avisos


# --------------------------------------------------------------------------- etapa completa


def preparar(cfg: Config, forcar: bool = False) -> dict:
    amostras, avisos = carregar_bruto(cfg)
    treino, val = dividir(amostras, cfg.divisao.proporcao_val, cfg.divisao.semente)
    if not val:
        raise ErroProjeto("Imagens de menos para separar validação. Junte mais fotos em dataset/bruto.")

    total = estatisticas(amostras, cfg.classes)
    if total["imagens_anotadas"] < cfg.divisao.minimo_imagens and not forcar:
        raise ErroProjeto(
            f"Só {total['imagens_anotadas']} imagens anotadas; a atividade pede pelo menos "
            f"{cfg.divisao.minimo_imagens}. Anote mais fotos (ou use --forcar para seguir só testando)."
        )
    avisos += avisos_balanceamento(total["instancias"])

    escrever_split(cfg, treino, val)
    data_yaml = escrever_data_yaml(cfg)
    ineditas = checar_ineditas(cfg, {a.hash: a.imagem.name for a in amostras})
    if len(ineditas) < cfg.divisao.minimo_ineditas:
        avisos.append(
            f"Há {len(ineditas)} imagem(ns) em {cfg.relativo(cfg.p('ineditas'))}; a atividade pede pelo "
            f"menos {cfg.divisao.minimo_ineditas} fotos novas para a detecção final."
        )

    proporcao_real = len(val) / len(amostras)
    sintetico = (cfg.p("bruto") / MARCADOR_SINTETICO).exists()
    resumo = {
        "sintetico": sintetico,
        "classes": cfg.classes,
        "data_yaml": cfg.relativo(data_yaml),
        "total": total,
        "treino": estatisticas(treino, cfg.classes),
        "validacao": estatisticas(val, cfg.classes),
        "proporcao_validacao": round(proporcao_real, 3),
        "semente": cfg.divisao.semente,
        "ineditas": {"quantidade": len(ineditas), "arquivos": [p.name for p in ineditas]},
        "avisos": avisos,
        "checklist": {
            "pelo_menos_2_classes": len(cfg.classes) >= 2,
            f"pelo_menos_{cfg.divisao.minimo_imagens}_imagens_anotadas": total["imagens_anotadas"]
            >= cfg.divisao.minimo_imagens,
            "divisao_proxima_de_80_20": abs(proporcao_real - cfg.divisao.proporcao_val) <= 0.07,
            "todas_as_classes_no_treino": all(v > 0 for v in estatisticas(treino, cfg.classes)["instancias"].values()),
            "todas_as_classes_na_validacao": all(
                v > 0 for v in estatisticas(val, cfg.classes)["instancias"].values()
            ),
            f"pelo_menos_{cfg.divisao.minimo_ineditas}_ineditas": len(ineditas) >= cfg.divisao.minimo_ineditas,
            "ineditas_sem_copia_no_treino_val": True,  # se houvesse cópia, checar_ineditas já teria parado
            "dataset_real_nao_sintetico": not sintetico,
        },
    }
    destino = cfg.p("resultados") / "dataset" / "resumo_dataset.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


# --------------------------------------------------------------------------- importação


def importar_pasta_yolo(cfg: Config, origem_pasta: Path, substituir: bool = False) -> dict:
    """Copia um export YOLO (Roboflow, CVAT, LabelImg...) para dataset/bruto.

    Junta train/valid/test do export numa pasta só, porque quem divide é o `preparar` (com semente e
    estratificação). Se o export tiver data.yaml, remapeia as classes pelo NOME.
    """
    origem_pasta = Path(origem_pasta)
    if not origem_pasta.is_dir():
        raise ErroProjeto(f"Pasta não encontrada: {origem_pasta}")

    nomes_origem = _nomes_do_export(origem_pasta)
    mapa: dict[int, int] | None = None
    if nomes_origem:
        faltando = [n for n in nomes_origem if n not in cfg.classes]
        if faltando:
            raise ErroProjeto(
                f"O export tem classes que não estão em configs/projeto.yaml: {faltando}. "
                f"Classes do export: {nomes_origem}. Renomeie no projeto.yaml ou na ferramenta de anotação."
            )
        mapa = {i: cfg.classes.index(n) for i, n in enumerate(nomes_origem)}

    destino_img = cfg.p("bruto") / "images"
    destino_rot = cfg.p("bruto") / "labels"
    if substituir:
        for pasta in (destino_img, destino_rot):
            if pasta.exists():
                shutil.rmtree(pasta)
    destino_img.mkdir(parents=True, exist_ok=True)
    destino_rot.mkdir(parents=True, exist_ok=True)

    copiadas = 0
    sem_imagem = 0
    for rotulo in sorted(origem_pasta.rglob("*.txt")):
        if rotulo.stem.lower() in {"classes", "readme"} or "labels" not in rotulo.parts:
            continue
        pasta_img = Path(*["images" if p == "labels" else p for p in rotulo.parent.parts])
        imagem = _achar_imagem(pasta_img, rotulo.stem)
        if imagem is None:
            sem_imagem += 1
            continue
        n_classes = len(nomes_origem) if nomes_origem else len(cfg.classes)
        caixas = ler_rotulo(rotulo, n_classes)
        if mapa is not None:
            caixas = [Caixa(mapa[c.classe], c.xc, c.yc, c.w, c.h) for c in caixas]
        nome = imagem.name
        if (destino_img / nome).exists() and not substituir:
            nome = f"{rotulo.parent.parent.name}_{nome}"  # mesmo nome em splits diferentes do export
        shutil.copy2(imagem, destino_img / nome)
        escrever_rotulo(destino_rot / f"{Path(nome).stem}.txt", caixas)
        copiadas += 1
    if copiadas == 0:
        raise ErroProjeto(
            f"Não achei pares images/ + labels/ dentro de {origem_pasta}. O export precisa estar no formato YOLO."
        )
    return {"copiadas": copiadas, "rotulos_sem_imagem": sem_imagem, "classes_do_export": nomes_origem}


def baixar_roboflow(cfg: Config, workspace: str, projeto: str, versao: int) -> dict:
    chave = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not chave:
        raise ErroProjeto(
            "Defina a variável de ambiente ROBOFLOW_API_KEY com sua chave (Roboflow > Settings > API Keys). "
            'PowerShell: $env:ROBOFLOW_API_KEY="sua_chave"   |   Linux: export ROBOFLOW_API_KEY=sua_chave'
        )
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise ErroProjeto('Pacote roboflow não instalado. Rode: pip install -e ".[roboflow]"') from exc
    destino = cfg.p("dataset") / "_download" / f"{projeto}-v{versao}"
    rf = Roboflow(api_key=chave)
    rf.workspace(workspace).project(projeto).version(versao).download("yolov8", location=str(destino))
    return importar_pasta_yolo(cfg, destino)


def _nomes_do_export(pasta: Path) -> list[str]:
    arquivo = next(iter(sorted(pasta.rglob("data.yaml"))), None)
    if arquivo is None:
        return []
    dados = yaml.safe_load(arquivo.read_text(encoding="utf-8")) or {}
    nomes = dados.get("names", [])
    if isinstance(nomes, dict):
        return [str(nomes[k]) for k in sorted(nomes)]
    return [str(n) for n in nomes]


def _achar_imagem(pasta: Path, stem: str) -> Path | None:
    for ext in sorted(EXTENSOES_IMAGEM):
        for variante in (ext, ext.upper()):
            candidato = pasta / f"{stem}{variante}"
            if candidato.is_file():
                return candidato
    return None
