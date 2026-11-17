"""Monta dataset/bruto a partir do "Orange Fruit Diseases Dataset" (Mendeley Data, CC BY 4.0).

Por que este script existe
--------------------------
O ZIP público do dataset traz só as imagens, separadas em pastas por diagnóstico
(healthy, Black spot, Canker, Greening, Scab). Os rótulos de caixa citados na página não vêm junto.
Então a anotação foi feita de forma SEMI-AUTOMÁTICA:

1. o YOLOv8n pré-treinado no COCO (classes "orange" e "apple") propõe a caixa da fruta;
2. a classe vem da pasta, que é o diagnóstico dado pelos autores do dataset:
   healthy -> laranja_boa ; Black spot / Canker / Greening / Scab -> laranja_defeito;
3. só fica a imagem com EXATAMENTE uma fruta detectada com confiança. Em foto de árvore com várias
   laranjas, o rótulo da pasta não vale para cada fruta, então ela sai;
4. quase-duplicatas (cópias aumentadas) saem por hash perceptual;
5. as classes são equilibradas e algumas imagens ficam separadas como inéditas, com gabarito;
6. tudo é reduzido para no máximo 640 px (o YOLO treina em 640 de qualquer jeito).

Uso (da raiz do projeto, com a .venv):
    python scripts/anotar_mendeley.py "<pasta>/Orange Fruit Diseases"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from citrus_yolo.dataset import Caixa, escrever_rotulo  # noqa: E402
from citrus_yolo.render import ler_imagem, salvar_imagem  # noqa: E402

PASTAS = {"healthy": 0, "Black spot": 1, "Canker": 1, "Greening": 1, "Scab": 1}
COCO_FRUTAS = [47, 49]  # apple, orange (laranja verde às vezes sai como "apple")
LADO_MAX = 640
FONTE = (
    "Doh, B. (2025). Orange Fruit Diseases Dataset (v1). Mendeley Data. "
    "https://doi.org/10.17632/6szsnpypdd.1 (licença CC BY 4.0)"
)


def dhash(img: np.ndarray) -> int:
    cinza = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (cinza[:, 1:] > cinza[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def caixa_por_cor(img: np.ndarray) -> list[float] | None:
    """Plano B para close-up em fundo liso (branco/preto), onde o detector do COCO costuma falhar.

    A fruta tem cor (saturação alta) e o fundo não. Só aceita quando pelo menos 3 dos 4 cantos
    são fundo sem cor (descarta fotos de folhagem) e existe UMA mancha dominante.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mascara = ((hsv[..., 1] > 60) & (hsv[..., 2] > 40)).astype(np.uint8)
    h, w = mascara.shape
    k = max(4, min(h, w) // 12)
    cantos = [mascara[:k, :k], mascara[:k, -k:], mascara[-k:, :k], mascara[-k:, -k:]]
    if sum(c.mean() < 0.15 for c in cantos) < 3:
        return None
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara = cv2.morphologyEx(cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo), cv2.MORPH_CLOSE, nucleo)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara)
    if n < 2:
        return None
    areas = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)), reverse=True)
    maior, idx = areas[0]
    if maior < 0.15 * h * w or (len(areas) > 1 and areas[1][0] > 0.3 * maior):
        return None
    x, y, bw, bh = stats[idx, :4]
    return [x / w, y / h, (x + bw) / w, (y + bh) / h]


def reduzir(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    escala = LADO_MAX / max(h, w)
    return cv2.resize(img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA) if escala < 1 else img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origem", type=Path, help="pasta 'Orange Fruit Diseases' extraída do ZIP")
    ap.add_argument("--conf", type=float, default=0.35, help="confiança mínima da caixa proposta")
    ap.add_argument("--area-min", type=float, default=0.05, help="área mínima da caixa (fração da imagem)")
    ap.add_argument("--ineditas-por-classe", type=int, default=4)
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()

    from ultralytics import YOLO

    pesos = RAIZ / "modelo" / "yolov8n.pt"
    detector = YOLO(str(pesos) if pesos.is_file() else "yolov8n.pt")
    aceitas, log = [], []
    for pasta, classe in PASTAS.items():
        for arquivo in sorted((args.origem / pasta).iterdir()):
            img = ler_imagem(arquivo)
            r = detector.predict(img, conf=0.2, classes=COCO_FRUTAS, verbose=False)[0]
            caixas = [(b.xyxyn[0].tolist(), float(b.conf[0])) for b in r.boxes if float(b.conf[0]) >= args.conf]
            motivo, metodo = "", "yolo_coco"
            if len(r.boxes) == 0:
                por_cor = caixa_por_cor(img)
                if por_cor is None:
                    motivo = "nenhuma fruta detectada"
                else:
                    caixas, metodo = [(por_cor, 1.0)], "cor_fundo_liso"
            elif len(r.boxes) > 2:
                motivo = "várias frutas na foto"
            elif len(caixas) != 1:
                motivo = "detecção ambígua"
            if not motivo:
                (x1, y1, x2, y2), _ = caixas[0]
                if (x2 - x1) * (y2 - y1) < args.area_min:
                    motivo = "fruta pequena demais na foto"
            log.append({"arquivo": f"{pasta}/{arquivo.name}", "classe_pasta": pasta, "frutas_propostas": len(r.boxes),
                        "metodo": metodo if not motivo else "", "aceita": not motivo, "motivo": motivo})
            if not motivo:
                aceitas.append({"arquivo": arquivo, "pasta": pasta, "classe": classe, "img": img, "metodo": metodo,
                                "caixa": caixas[0][0], "conf": caixas[0][1], "hash": dhash(img)})

    # quase-duplicatas: distância de Hamming pequena no dHash = mesma foto (ou quase)
    unicas = []
    for a in aceitas:
        if all(bin(a["hash"] ^ u["hash"]).count("1") > 5 for u in unicas):
            unicas.append(a)
    removidas_dup = len(aceitas) - len(unicas)

    rng = random.Random(args.semente)
    por_classe = defaultdict(list)
    for a in unicas:
        por_classe[a["classe"]].append(a)
    boas = por_classe[0]
    # defeito: rodízio entre as 4 doenças até igualar o número de boas (mantém a variedade)
    por_doenca = defaultdict(list)
    for a in por_classe[1]:
        por_doenca[a["pasta"]].append(a)
    for lista in por_doenca.values():
        rng.shuffle(lista)
    defeitos = []
    while len(defeitos) < len(boas) and any(por_doenca.values()):
        for doenca in sorted(por_doenca):
            if por_doenca[doenca] and len(defeitos) < len(boas):
                defeitos.append(por_doenca[doenca].pop())
    rng.shuffle(boas)

    ineditas = boas[: args.ineditas_por_classe] + defeitos[: args.ineditas_por_classe]
    treino_val = boas[args.ineditas_por_classe :] + defeitos[args.ineditas_por_classe :]

    bruto = RAIZ / "dataset" / "bruto"
    pasta_ineditas = RAIZ / "dataset" / "ineditas"
    for alvo in (bruto / "images", bruto / "labels", pasta_ineditas / "rotulos"):
        if alvo.exists():
            shutil.rmtree(alvo)
    for f in pasta_ineditas.glob("*.jpg"):
        f.unlink()

    def salvar(a: dict, pasta_img: Path, pasta_rot: Path) -> str:
        nome = f"{a['pasta'].lower().replace(' ', '_')}_{a['arquivo'].stem}.jpg"
        salvar_imagem(pasta_img / nome, reduzir(a["img"]), qualidade=90)
        x1, y1, x2, y2 = a["caixa"]
        caixa = Caixa(a["classe"], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        escrever_rotulo(pasta_rot / f"{Path(nome).stem}.txt", [caixa])
        return nome

    for a in treino_val:
        salvar(a, bruto / "images", bruto / "labels")
    for a in ineditas:
        salvar(a, pasta_ineditas, pasta_ineditas / "rotulos")

    with open(bruto / "anotacao_semiautomatica.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    motivos = Counter(item["motivo"] for item in log if item["motivo"])
    resumo = {
        "fonte": FONTE,
        "metodo": "caixa proposta pelo YOLOv8n (COCO: orange/apple) ou, em fundo liso, por segmentação de cor; "
                  "classe da pasta do dataset; revisão visual por amostragem",
        "caixas_por_metodo": dict(Counter(a["metodo"] for a in treino_val + ineditas)),
        "imagens_no_zip": len(log),
        "aceitas_com_1_fruta": len(aceitas),
        "descartadas": dict(motivos),
        "quase_duplicatas_removidas": removidas_dup,
        "treino_val": {"laranja_boa": sum(a["classe"] == 0 for a in treino_val),
                       "laranja_defeito": sum(a["classe"] == 1 for a in treino_val)},
        "defeito_por_doenca": dict(Counter(a["pasta"] for a in defeitos[args.ineditas_por_classe :])),
        "ineditas": len(ineditas),
        "confianca_minima_caixa": args.conf,
        "semente": args.semente,
    }
    (bruto / "ORIGEM.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
