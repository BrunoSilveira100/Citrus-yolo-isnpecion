"""Treino e validação com o Ultralytics.

O `ultralytics` (e o torch) só é importado dentro das funções. Assim o resto do projeto (preparar
dataset, relatório, testes) funciona mesmo numa máquina onde o torch não carrega.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

from .config import EPOCAS_MINIMAS, Config, ErroProjeto

# Arquivos que o Ultralytics gera e que valem como evidência no relatório.
GRAFICOS_TREINO = [
    "results.png",
    "results.csv",
    "args.yaml",
    "labels.jpg",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "*PR_curve.png",
    "*F1_curve.png",
    "*P_curve.png",
    "*R_curve.png",
    "train_batch*.jpg",
    "val_batch*_labels.jpg",
    "val_batch*_pred.jpg",
]

COLUNAS_EPOCA = [
    ("epoch", "epoca"),
    ("train/box_loss", "box_loss"),
    ("train/cls_loss", "cls_loss"),
    ("train/dfl_loss", "dfl_loss"),
    ("metrics/precision(B)", "precision"),
    ("metrics/recall(B)", "recall"),
    ("metrics/mAP50(B)", "mAP50"),
    ("metrics/mAP50-95(B)", "mAP50_95"),
]


def carregar_yolo():
    """Importa a classe YOLO com mensagens claras para os dois problemas mais comuns."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ErroProjeto('Ultralytics não está instalado. Rode: pip install -e ".[dev]"') from exc
    except OSError as exc:
        raise ErroProjeto(
            f"O PyTorch não conseguiu carregar nesta máquina ({exc}). No Windows isso costuma ser o "
            "Smart App Control / Controle de Aplicativo bloqueando as DLLs do torch. Saídas: treinar pelo "
            "notebooks/treino_colab.ipynb (GPU grátis) ou usar outra máquina."
        ) from exc
    return YOLO


# --------------------------------------------------------------------------- treino


def treinar(cfg: Config, comando: str = "") -> dict:
    t = cfg.treino
    if t.epocas < EPOCAS_MINIMAS and not t.justificativa_epocas.strip():
        raise ErroProjeto(f"Menos de {EPOCAS_MINIMAS} épocas exige --justificativa.")
    if not cfg.data_yaml.is_file():
        raise ErroProjeto("Não achei dataset/data.yaml. Rode antes: citrus-yolo preparar")

    YOLO = carregar_yolo()
    pasta_saida = cfg.p("resultados") / "treino"
    if pasta_saida.exists():
        shutil.rmtree(pasta_saida)
    pasta_saida.mkdir(parents=True)

    log = logging.FileHandler(pasta_saida / "log_treino.txt", encoding="utf-8")
    log.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("ultralytics")
    logger.addHandler(log)
    inicio = time.perf_counter()
    # pesos base guardados em modelo/ (ex.: modelo/yolov8n.pt); se não existirem, o Ultralytics baixa
    local = cfg.p("modelo") / t.modelo_base
    base = str(local) if local.is_file() else t.modelo_base
    try:
        modelo = YOLO(base)
        modelo.train(
            data=str(cfg.data_yaml),
            epochs=t.epocas,
            imgsz=t.imgsz,
            batch=t.batch,
            device=t.dispositivo,
            patience=t.paciencia,
            seed=cfg.divisao.semente,
            project=str(cfg.p("runs")),
            name=t.nome_execucao,
            plots=True,
        )
    finally:
        logger.removeHandler(log)
        log.close()
    duracao = time.perf_counter() - inicio

    pasta_run = Path(modelo.trainer.save_dir)
    melhor = pasta_run / "weights" / "best.pt"
    if not melhor.is_file():
        raise ErroProjeto(f"O treino terminou mas não gerou {melhor}. Veja o log em {pasta_saida}.")
    cfg.best_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(melhor, cfg.best_pt)
    copiar_graficos(pasta_run, pasta_saida, GRAFICOS_TREINO)

    linhas = tabela_epocas(pasta_run / "results.csv")
    (pasta_saida / "epocas.txt").write_text(formatar_epocas(linhas), encoding="utf-8")
    melhor_epoca = max(linhas, key=lambda x: x.get("mAP50_95", 0.0)) if linhas else {}
    info = {
        "modelo_base": t.modelo_base,
        "epocas_pedidas": t.epocas,
        "epocas_executadas": len(linhas),
        "justificativa_epocas": t.justificativa_epocas,
        "imgsz": t.imgsz,
        "batch": t.batch,
        "dispositivo": t.dispositivo,
        "duracao_s": round(duracao, 1),
        "pasta_run": cfg.relativo(pasta_run),
        "best_pt_original": cfg.relativo(melhor),
        "best_pt": cfg.relativo(cfg.best_pt),
        "best_pt_mb": round(cfg.best_pt.stat().st_size / 1e6, 2),
        "melhor_epoca": melhor_epoca,
        "ultima_epoca": linhas[-1] if linhas else {},
    }
    (pasta_saida / "treino.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def tabela_epocas(arquivo_csv: Path) -> list[dict]:
    """Lê o results.csv do Ultralytics e devolve só as colunas que interessam, com nomes curtos."""
    if not Path(arquivo_csv).is_file():
        return []
    with open(arquivo_csv, encoding="utf-8") as fh:
        leitor = csv.DictReader(fh)
        linhas = []
        for bruta in leitor:
            limpa = {k.strip(): v.strip() for k, v in bruta.items() if k}
            linha = {}
            for original, curto in COLUNAS_EPOCA:
                if original in limpa and limpa[original] != "":
                    valor = float(limpa[original])
                    linha[curto] = int(valor) if curto == "epoca" else round(valor, 4)
            linhas.append(linha)
    return linhas


def formatar_epocas(linhas: list[dict]) -> str:
    if not linhas:
        return "Sem dados de épocas (results.csv não encontrado).\n"
    colunas = [curto for _, curto in COLUNAS_EPOCA if any(curto in linha for linha in linhas)]
    largura = 11
    saida = ["".join(c.rjust(largura) for c in colunas)]
    for linha in linhas:
        celulas = []
        for c in colunas:
            valor = linha.get(c, "")
            celulas.append((str(valor) if isinstance(valor, int) else f"{valor:.4f}" if valor != "" else "-").rjust(largura))
        saida.append("".join(celulas))
    return "\n".join(saida) + "\n"


def copiar_graficos(origem: Path, destino: Path, padroes: list[str]) -> list[str]:
    destino.mkdir(parents=True, exist_ok=True)
    copiados = []
    for padrao in padroes:
        for arquivo in sorted(Path(origem).glob(padrao)):
            if arquivo.is_file() and arquivo.name not in copiados:
                shutil.copy2(arquivo, destino / arquivo.name)
                copiados.append(arquivo.name)
    return copiados


def info_treino(cfg: Config) -> dict | None:
    arquivo = cfg.p("resultados") / "treino" / "treino.json"
    return json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.is_file() else None


# --------------------------------------------------------------------------- validação


def validar(cfg: Config, pesos: Path | None = None) -> dict:
    pesos = Path(pesos) if pesos else cfg.best_pt
    if not pesos.is_file():
        raise ErroProjeto(f"Não achei {cfg.relativo(pesos)}. Rode antes: citrus-yolo treinar")
    if not cfg.data_yaml.is_file():
        raise ErroProjeto("Não achei dataset/data.yaml. Rode antes: citrus-yolo preparar")

    YOLO = carregar_yolo()
    modelo = YOLO(str(pesos))
    resultado = modelo.val(
        data=str(cfg.data_yaml),
        imgsz=cfg.treino.imgsz,
        batch=cfg.treino.batch,
        device=cfg.treino.dispositivo,
        split="val",
        project=str(cfg.p("runs")),
        name="val",
        exist_ok=True,
        plots=True,
    )
    caixa = resultado.box
    nomes = modelo.names
    geral = {
        "precision": float(caixa.mp),
        "recall": float(caixa.mr),
        "mAP50": float(caixa.map50),
        "mAP50_95": float(caixa.map),
    }
    por_classe = {}
    for i, indice in enumerate(caixa.ap_class_index):
        por_classe[nomes[int(indice)]] = {
            "precision": float(caixa.p[i]),
            "recall": float(caixa.r[i]),
            "mAP50": float(caixa.ap50[i]),
            "mAP50_95": float(caixa.ap[i]),
        }
    metricas = montar_metricas(geral, por_classe, list(nomes.values()), cfg.relativo(pesos))

    pasta = cfg.p("resultados") / "validacao"
    if pasta.exists():
        shutil.rmtree(pasta)
    copiar_graficos(Path(resultado.save_dir), pasta, GRAFICOS_TREINO)
    (pasta / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / "metricas.md").write_text(metricas_markdown(metricas), encoding="utf-8")
    return metricas


def montar_metricas(geral: dict, por_classe: dict, classes: list[str], pesos: str) -> dict:
    return {
        "pesos": pesos,
        "geral": {k: round(v, 4) for k, v in geral.items()},
        "por_classe": {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in por_classe.items()},
        "classes_sem_instancia_na_validacao": [c for c in classes if c not in por_classe],
    }


EXPLICACAO_CURTA = {
    "precision": "das caixas que o modelo desenhou, quantas estavam certas (poucos alarmes falsos).",
    "recall": "das laranjas que existiam, quantas o modelo encontrou (poucas frutas escapando).",
    "mAP50": "nota geral de 0 a 1 contando acerto quando a caixa cobre >= 50% da real (IoU 0,5).",
    "mAP50_95": "mesma nota, mais exigente: média de IoU 0,50 a 0,95 (caixa bem ajustada).",
}


def metricas_markdown(m: dict) -> str:
    linhas = [
        "# Métricas de validação",
        "",
        f"Pesos avaliados: `{m['pesos']}`",
        "",
        "| Conjunto | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    def linha(nome: str, v: dict) -> str:
        return f"| {nome} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} |"

    linhas.append(linha("**todas as classes**", m["geral"]))
    for nome, valores in m["por_classe"].items():
        linhas.append(linha(nome, valores))
    if m["classes_sem_instancia_na_validacao"]:
        linhas += ["", f"Sem exemplos na validação: {', '.join(m['classes_sem_instancia_na_validacao'])}."]
    linhas += ["", "## Como ler", ""]
    linhas += [f"- **{k}**: {v}" for k, v in EXPLICACAO_CURTA.items()]
    return "\n".join(linhas) + "\n"
