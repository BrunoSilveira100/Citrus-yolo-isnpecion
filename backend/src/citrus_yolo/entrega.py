"""Monta o ZIP da entrega com a estrutura pedida no PDF da atividade.

Projeto_Final_YOLO_<Nome>/
├── dataset/        A: images/, labels/, ineditas/, data.yaml (sem `path`, portável)
├── modelo/best.pt  B
├── resultados/     C: treino, validação, detecções
├── relatorio.pdf   D
├── comandos.txt    E
└── codigo/         src, configs, tests, notebooks, docs, README, pyproject
"""

from __future__ import annotations

import unicodedata
import zipfile
from pathlib import Path

from .config import Config
from .dataset import MARCADOR_SINTETICO, conteudo_data_yaml, listar_imagens
from .registro import NOME_ARQUIVO as LOG_COMANDOS
from .relatorio import carregar_textos, pendencias_textos

LIMITE_MB = 100
# Relativos a backend/ (a raiz da config). O ZIP leva o repositório inteiro em codigo/,
# mantendo a mesma divisão backend/ + frontend/.
CODIGO = ["src", "scripts", "configs", "tests", "pyproject.toml", "requirements.txt", "comandos.txt",
          "AGENTS.md", "CLAUDE.md"]
REPOSITORIO = ["README.md", "frontend", ".github"]  # relativos à pasta acima de backend/
IGNORAR = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def slug(nome: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    partes = "".join(c if c.isalnum() else " " for c in sem_acento).split()
    return "_".join(partes) or "SeuNome"


def pendencias(cfg: Config) -> list[str]:
    r = cfg.p("resultados")
    itens = []
    if (cfg.p("bruto") / MARCADOR_SINTETICO).exists():
        itens.append("O dataset em uso é SINTÉTICO. Troque por imagens reais (citrus-yolo sintetico --limpar).")
    if not cfg.aluno.nome.strip():
        itens.append("Preencha aluno.nome em configs/projeto.yaml.")
    if not cfg.data_yaml.is_file():
        itens.append("Dataset não preparado: rode citrus-yolo preparar.")
    if not cfg.best_pt.is_file():
        itens.append("modelo/best.pt não existe: rode citrus-yolo treinar.")
    if not (r / "validacao" / "metricas.json").is_file():
        itens.append("Métricas de validação faltando: rode citrus-yolo validar.")
    paineis = listar_imagens(r / "deteccoes" / "painel")
    if len(paineis) < cfg.divisao.minimo_ineditas:
        itens.append(f"Só {len(paineis)} imagens com detecção; a atividade pede {cfg.divisao.minimo_ineditas}+.")
    if not (r / "relatorio" / "relatorio.pdf").is_file():
        itens.append("relatorio.pdf faltando: rode citrus-yolo relatorio.")
    textos = pendencias_textos(carregar_textos(cfg))
    if textos:
        itens.append(f"configs/relatorio.yaml ainda tem rascunho 'EDITE:' em: {', '.join(textos)}.")
    if not listar_imagens(r / "prints"):
        itens.append("Coloque as capturas de tela do treino em resultados/prints/ (o PDF pede print da tela).")
    return itens


def _arquivos(pasta: Path):
    for p in sorted(pasta.rglob("*")):
        if p.is_file() and not (IGNORAR & set(p.parts)) and not p.name.endswith(".egg-info"):
            yield p


def montar_comandos(cfg: Config) -> str:
    base = cfg.raiz / "comandos.txt"
    texto = base.read_text(encoding="utf-8") if base.is_file() else ""
    log = cfg.p("resultados") / LOG_COMANDOS
    if log.is_file():
        texto += "\n\n" + "=" * 78 + "\nCOMANDOS EFETIVAMENTE EXECUTADOS (log automático, com data e hora)\n"
        texto += "=" * 78 + "\n\n" + log.read_text(encoding="utf-8")
    return texto


def entregar(cfg: Config) -> dict:
    nome = f"Projeto_Final_YOLO_{slug(cfg.aluno.nome)}"
    destino = cfg.p("entrega") / f"{nome}.zip"
    destino.parent.mkdir(parents=True, exist_ok=True)
    ds = cfg.p("dataset")
    r = cfg.p("resultados")

    with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # A. dataset (sem o bruto: as mesmas fotos já estão em images/ e labels/)
        for sub in ("images", "labels"):
            if (ds / sub).is_dir():
                for arq in _arquivos(ds / sub):
                    z.write(arq, f"{nome}/dataset/{arq.relative_to(ds).as_posix()}")
        if cfg.p("ineditas").is_dir():
            for arq in _arquivos(cfg.p("ineditas")):
                z.write(arq, f"{nome}/dataset/ineditas/{arq.relative_to(cfg.p('ineditas')).as_posix()}")
        z.writestr(f"{nome}/dataset/data.yaml", conteudo_data_yaml(cfg.classes, None))
        # documentação da origem/anotação do dataset (ORIGEM.json, CSV da anotação, etc.)
        for arq in sorted(cfg.p("bruto").glob("*")):
            if arq.is_file() and arq.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
                z.write(arq, f"{nome}/dataset/{arq.name}")
        # B. modelo
        if cfg.best_pt.is_file():
            z.write(cfg.best_pt, f"{nome}/modelo/best.pt")
        # C. evidências (o PDF vai na raiz do ZIP, não precisa duplicar)
        if r.is_dir():
            for arq in _arquivos(r):
                if arq.parent.name == "relatorio" and arq.suffix == ".pdf":
                    continue
                z.write(arq, f"{nome}/resultados/{arq.relative_to(r).as_posix()}")
        # D. relatório
        pdf = r / "relatorio" / "relatorio.pdf"
        if pdf.is_file():
            z.write(pdf, f"{nome}/relatorio.pdf")
        # E. comandos
        z.writestr(f"{nome}/comandos.txt", montar_comandos(cfg))
        # código-fonte: backend/ e, se existir, o resto do repositório (README, frontend/, CI)
        repo = cfg.raiz.parent
        prefixo = f"{cfg.raiz.name}/" if (repo / "frontend").is_dir() else ""
        itens = [(cfg.raiz / i, f"{prefixo}{i}") for i in CODIGO]
        if prefixo:
            itens += [(repo / i, i) for i in REPOSITORIO]
        for alvo, destino_rel in itens:
            if alvo.is_file():
                z.write(alvo, f"{nome}/codigo/{destino_rel}")
            elif alvo.is_dir():
                for arq in _arquivos(alvo):
                    z.write(arq, f"{nome}/codigo/{destino_rel}/{arq.relative_to(alvo).as_posix()}")

    tamanho_mb = destino.stat().st_size / 1e6
    avisos = pendencias(cfg)
    if tamanho_mb > LIMITE_MB:
        avisos.append(
            f"O ZIP tem {tamanho_mb:.0f} MB. Se o envio não aceitar, suba o dataset (Drive/Roboflow), coloque o "
            "link em configs/relatorio.yaml (link_dataset) e gere o relatório de novo."
        )
    return {"zip": cfg.relativo(destino), "tamanho_mb": round(tamanho_mb, 1), "pendencias": avisos}
