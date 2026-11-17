"""Detecção nas imagens inéditas + mosaico com exemplos anotados do dataset.

Para cada foto inédita sai:
  - o visual padrão do YOLO (caixa, classe, confiança);
  - um PAINEL DE INSPEÇÃO: a foto com as caixas + barra lateral com contagem e taxa de defeito;
  - se existir gabarito (ineditas/rotulos/<foto>.txt), a comparação acerto/erro automática.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from .config import Config, ErroProjeto
from .dataset import (
    hashes_do_split,
    ler_rotulo,
    listar_imagens,
    rotulo_da_inedita,
    sha1,
)
from .render import CaixaDesenho, desenhar_caixas, ler_imagem, mosaico, painel_inspecao, salvar_imagem


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def comparar(gabarito: list[CaixaDesenho], previstas: list[CaixaDesenho], iou_minimo: float = 0.5) -> dict:
    """Casa previsões com o gabarito (a mais confiante primeiro) e classifica cada caso.

    - acerto: caixa no lugar certo (IoU >= 0,5) e classe certa;
    - classe_trocada: caixa no lugar certo, mas o modelo errou boa x defeito;
    - falso_positivo: caixa onde não havia laranja;
    - nao_detectada: laranja do gabarito que ninguém encontrou.
    """
    livres = list(range(len(gabarito)))
    resultado = Counter()
    for prevista in sorted(previstas, key=lambda c: -(c.confianca or 0)):
        melhor, melhor_iou = None, iou_minimo
        for i in livres:
            valor = iou(prevista.xyxy, gabarito[i].xyxy)
            if valor >= melhor_iou:
                melhor, melhor_iou = i, valor
        if melhor is None:
            resultado["falso_positivo"] += 1
            continue
        livres.remove(melhor)
        resultado["acerto" if gabarito[melhor].classe == prevista.classe else "classe_trocada"] += 1
    resultado["nao_detectada"] += len(livres)
    saida = {k: resultado.get(k, 0) for k in ("acerto", "classe_trocada", "falso_positivo", "nao_detectada")}
    saida["situacao"] = "acerto" if sum(saida.values()) == saida["acerto"] else "erro"
    return saida


def detectar(
    cfg: Config,
    pesos: Path | None = None,
    fonte: Path | None = None,
    confianca: float | None = None,
    imgsz: int | None = None,
) -> dict:
    pesos = Path(pesos) if pesos else cfg.best_pt
    if not pesos.is_file():
        raise ErroProjeto(f"Não achei {cfg.relativo(pesos)}. Rode antes: citrus-yolo treinar")
    fonte = Path(fonte) if fonte else cfg.p("ineditas")
    imagens = listar_imagens(fonte)
    if not imagens:
        raise ErroProjeto(f"Nenhuma imagem em {cfg.relativo(fonte)}. Coloque lá as fotos inéditas.")

    conhecidos = hashes_do_split(cfg)
    vazadas = [img.name for img in imagens if sha1(img) in conhecidos]
    if vazadas:
        raise ErroProjeto(
            f"Imagens 'inéditas' que já estão no treino/validação: {', '.join(vazadas)}. Troque por fotos novas."
        )

    from .treino import carregar_yolo

    conf = confianca if confianca is not None else cfg.deteccao.confianca
    YOLO = carregar_yolo()
    modelo = YOLO(str(pesos))
    nomes = {int(k): v for k, v in modelo.names.items()}
    resultados = modelo.predict(
        source=[str(p) for p in imagens],
        conf=conf,
        iou=cfg.deteccao.iou,
        imgsz=imgsz or cfg.treino.imgsz,
        device=cfg.treino.dispositivo,
        save=True,
        project=str(cfg.p("runs")),
        name="predict",
        exist_ok=True,
        verbose=False,
    )

    saida = cfg.p("resultados") / "deteccoes"
    if saida.exists():
        shutil.rmtree(saida)
    (saida / "yolo").mkdir(parents=True)

    registros, paineis, linhas_csv = [], [], []
    for imagem_path, r in zip(imagens, resultados, strict=True):
        previstas = [
            CaixaDesenho(tuple(float(v) for v in xyxy), int(c), nomes[int(c)], float(p))
            for xyxy, c, p in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), strict=True
            )
        ]
        original = r.orig_img
        visual_yolo = Path(r.save_dir) / imagem_path.name
        if visual_yolo.is_file():
            shutil.copy2(visual_yolo, saida / "yolo" / imagem_path.name)
        else:
            salvar_imagem(saida / "yolo" / imagem_path.name, r.plot())

        contagem = {nome: sum(1 for c in previstas if c.nome == nome) for nome in nomes.values()}
        avaliacao = None
        gabarito_path = rotulo_da_inedita(imagem_path)
        if gabarito_path is not None:
            h, w = original.shape[:2]
            gabarito = [
                CaixaDesenho(c.xyxy(w, h), c.classe, nomes.get(c.classe, str(c.classe)))
                for c in ler_rotulo(gabarito_path, len(nomes))
            ]
            avaliacao = comparar(gabarito, previstas)

        rodape = f"conf >= {conf}"
        if avaliacao:
            rodape += f" | {avaliacao['situacao'].upper()}"
        painel = painel_inspecao(
            desenhar_caixas(original, previstas), contagem, cfg.deteccao.classe_defeito, rodape=rodape
        )
        salvar_imagem(saida / "painel" / f"{imagem_path.stem}.jpg", painel)
        paineis.append(painel)

        for c in previstas:
            linhas_csv.append([imagem_path.name, c.nome, f"{c.confianca:.4f}", *(f"{v:.1f}" for v in c.xyxy)])
        registros.append(
            {
                "arquivo": imagem_path.name,
                "deteccoes": [
                    {"classe": c.nome, "confianca": round(c.confianca, 4), "xyxy": [round(v, 1) for v in c.xyxy]}
                    for c in previstas
                ],
                "contagem": contagem,
                "avaliacao": avaliacao,
            }
        )

    with open(saida / "deteccoes.csv", "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["imagem", "classe", "confianca", "x1", "y1", "x2", "y2"])
        escritor.writerows(linhas_csv)
    salvar_imagem(saida / "mosaico_deteccoes.jpg", mosaico(paineis, colunas=2, largura_celula=560))

    resumo = {
        "pesos": cfg.relativo(pesos),
        "fonte": cfg.relativo(fonte),
        "confianca_minima": conf,
        "imagens": registros,
        "total_imagens": len(registros),
        "total_deteccoes": len(linhas_csv),
        "imagens_sem_deteccao": [r["arquivo"] for r in registros if not r["deteccoes"]],
        "com_gabarito": sum(1 for r in registros if r["avaliacao"]),
    }
    (saida / "resumo_deteccoes.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def exemplos_anotados(cfg: Config, quantidade: int = 6) -> Path | None:
    """Mosaico com fotos do treino e as caixas ANOTADAS (o que o modelo vai aprender)."""
    pasta_img = cfg.p("dataset") / "images" / "train"
    pasta_rot = cfg.p("dataset") / "labels" / "train"
    imagens = listar_imagens(pasta_img)
    if not imagens:
        return None
    # tenta mostrar todas as classes: primeiro uma foto de cada, depois completa
    escolhidas: list[Path] = []
    for indice in range(len(cfg.classes)):
        for img in imagens:
            rot = pasta_rot / f"{img.stem}.txt"
            if img not in escolhidas and any(c.classe == indice for c in ler_rotulo(rot, len(cfg.classes))):
                escolhidas.append(img)
                break
    escolhidas += [i for i in imagens if i not in escolhidas][: max(0, quantidade - len(escolhidas))]

    quadros = []
    for img in escolhidas[:quantidade]:
        foto = ler_imagem(img)
        h, w = foto.shape[:2]
        caixas = [
            CaixaDesenho(c.xyxy(w, h), c.classe, cfg.classes[c.classe])
            for c in ler_rotulo(pasta_rot / f"{img.stem}.txt", len(cfg.classes))
        ]
        quadros.append(desenhar_caixas(foto, caixas))
    destino = cfg.p("resultados") / "dataset" / "exemplos_anotados.jpg"
    return salvar_imagem(destino, mosaico(quadros, colunas=3))
