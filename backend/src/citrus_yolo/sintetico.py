"""Gera laranjas sintéticas sobre uma esteira, SÓ para testar o pipeline de ponta a ponta.

Não serve para a entrega: o professor quer imagens reais (dataset público ou fotos suas).
Por isso as pastas geradas recebem um SINTETICO.txt, e o relatório e o `entregar` avisam em
vermelho enquanto ele existir.
"""

from __future__ import annotations

import shutil

import cv2
import numpy as np

from .config import Config, ErroProjeto
from .dataset import MARCADOR_SINTETICO, Caixa, escrever_rotulo, listar_imagens
from .render import salvar_imagem

AVISO = (
    "Pasta gerada por `citrus-yolo sintetico`. Imagens ARTIFICIAIS, só para testar o pipeline.\n"
    "NÃO use na entrega. Para limpar: citrus-yolo sintetico --limpar\n"
)


def gerar(cfg: Config, n_imagens: int = 80, n_ineditas: int = 6, semente: int = 7) -> dict:
    bruto = cfg.p("bruto")
    ineditas = cfg.p("ineditas")
    _garantir_que_nao_tem_dado_real(bruto, listar_imagens(bruto / "images"))
    _garantir_que_nao_tem_dado_real(ineditas, listar_imagens(ineditas))
    _limpar_pasta_sintetica(bruto)
    _limpar_pasta_sintetica(ineditas)

    rng = np.random.default_rng(semente)
    contagem = [0] * len(cfg.classes)
    for i in range(n_imagens):
        imagem, caixas = cena(rng)
        salvar_imagem(bruto / "images" / f"sint_{i:03d}.jpg", imagem)
        escrever_rotulo(bruto / "labels" / f"sint_{i:03d}.txt", caixas)
        for c in caixas:
            contagem[c.classe] += 1
    for i in range(n_ineditas):
        imagem, caixas = cena(rng)
        salvar_imagem(ineditas / f"sint_inedita_{i:02d}.jpg", imagem)
        escrever_rotulo(ineditas / "rotulos" / f"sint_inedita_{i:02d}.txt", caixas)

    for pasta in (bruto, ineditas):
        (pasta / MARCADOR_SINTETICO).write_text(AVISO, encoding="utf-8")
    return {
        "imagens": n_imagens,
        "ineditas": n_ineditas,
        "instancias": {cfg.classes[i]: n for i, n in enumerate(contagem) if i < 2},
    }


def limpar(cfg: Config) -> list[str]:
    """Remove o dataset sintético e tudo o que foi gerado a partir dele."""
    bruto = cfg.p("bruto")
    if not (bruto / MARCADOR_SINTETICO).exists():
        raise ErroProjeto("Não há dataset sintético em uso (sem SINTETICO.txt em dataset/bruto). Nada a limpar.")
    removidos = []
    for pasta in (bruto, cfg.p("ineditas")):
        if (pasta / MARCADOR_SINTETICO).exists():
            _limpar_pasta_sintetica(pasta)
            removidos.append(cfg.relativo(pasta))
    gerados = [
        cfg.p("dataset") / "images",
        cfg.p("dataset") / "labels",
        cfg.data_yaml,
        cfg.best_pt,
        cfg.p("runs"),
        cfg.p("entrega"),
    ]
    # resultados/ sai inteiro, menos prints/ (onde ficam as capturas de tela reais)
    if cfg.p("resultados").is_dir():
        gerados += [p for p in cfg.p("resultados").iterdir() if p.name != "prints"]
    for alvo in gerados:
        if alvo.is_dir():
            shutil.rmtree(alvo)
            removidos.append(cfg.relativo(alvo))
        elif alvo.is_file():
            alvo.unlink()
            removidos.append(cfg.relativo(alvo))
    return removidos


# --------------------------------------------------------------------------- desenho da cena


def cena(rng: np.random.Generator, largura: int = 640, altura: int = 480) -> tuple[np.ndarray, list[Caixa]]:
    img = _esteira(rng, largura, altura)
    margem_trilho = 20
    laranjas = []
    for _ in range(int(rng.integers(1, 5))):
        for _tentativa in range(40):
            rx = int(rng.integers(int(largura * 0.06), int(largura * 0.112)))  # 38..72 px numa foto de 640
            ry = int(rx * rng.uniform(0.9, 1.05))
            cx = int(rng.integers(rx + 4, largura - rx - 4))
            cy = int(rng.integers(margem_trilho + ry + 4, altura - margem_trilho - ry - 4))
            if all(np.hypot(cx - x, cy - y) > rx + r + 6 for x, y, r, _, _ in laranjas):
                laranjas.append((cx, cy, rx, ry, bool(rng.random() < 0.5)))
                break

    for cx, cy, rx, ry, _ in laranjas:  # sombras primeiro, para ficarem por baixo de todas as frutas
        cv2.ellipse(img, (cx + 7, cy + 9), (rx, ry), 0, 0, 360, (20, 20, 20), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (15, 15), 0)

    caixas = []
    for cx, cy, rx, ry, defeito in laranjas:
        _laranja(img, rng, cx, cy, rx, ry, defeito)
        caixas.append(Caixa(int(defeito), cx / largura, cy / altura, 2 * rx / largura, 2 * ry / altura))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, caixas


def _esteira(rng: np.random.Generator, largura: int, altura: int) -> np.ndarray:
    base = int(rng.integers(45, 80))
    img = np.full((altura, largura, 3), base, np.int16)
    passo = int(rng.integers(16, 30))
    for y in range(0, altura, passo):
        img[y : y + 2] += 14
    img += rng.normal(0, 6, img.shape).astype(np.int16)
    luz = np.linspace(-25, 25, largura)[None, :, None] * rng.choice([-1, 1])
    img = np.clip(img + luz, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (0, 0), (largura, 18), (95, 95, 100), -1)
    cv2.rectangle(img, (0, altura - 18), (largura, altura), (95, 95, 100), -1)
    return img


def _laranja(img: np.ndarray, rng: np.random.Generator, cx: int, cy: int, rx: int, ry: int, defeito: bool) -> None:
    cor = (int(rng.integers(0, 30)), int(rng.integers(120, 170)), int(rng.integers(225, 256)))  # BGR
    angulo = float(rng.uniform(0, 180))
    cv2.ellipse(img, (cx, cy), (rx, ry), angulo, 0, 360, cor, -1, cv2.LINE_AA)
    escura = tuple(max(0, c - 35) for c in cor)
    for _ in range(int(rx * ry / 35)):  # poros da casca
        a, r = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 0.85))
        cv2.circle(img, (int(cx + r * rx * np.cos(a)), int(cy + r * ry * np.sin(a))), 1, escura, -1)
    clara = tuple(min(255, c + 60) for c in cor)
    cv2.ellipse(img, (cx - rx // 3, cy - ry // 3), (rx // 4, ry // 5), -30, 0, 360, clara, -1, cv2.LINE_AA)
    cv2.circle(img, (cx + int(rng.integers(-rx // 4, rx // 4 + 1)), cy - ry + 7), 4, (30, 95, 40), -1)

    if not defeito:
        return
    for _ in range(int(rng.integers(1, 4))):
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.55)
        px, py = cx + r * rx * np.cos(a), cy + r * ry * np.sin(a)
        raio = rx * rng.uniform(0.14, 0.34)
        angulos = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        raios = raio * rng.uniform(0.6, 1.25, len(angulos))
        poligono = np.stack([px + raios * np.cos(angulos), py + raios * np.sin(angulos)], axis=1).astype(np.int32)
        if rng.random() < 0.6:  # mancha / podridão
            cv2.fillPoly(img, [poligono], (20, 45, 70), cv2.LINE_AA)
            cv2.circle(img, (int(px), int(py)), max(2, int(raio * 0.4)), (15, 25, 40), -1, cv2.LINE_AA)
        else:  # bolor
            cv2.fillPoly(img, [poligono], (175, 205, 170), cv2.LINE_AA)
            cv2.circle(img, (int(px), int(py)), max(2, int(raio * 0.35)), (110, 150, 110), -1, cv2.LINE_AA)


def _garantir_que_nao_tem_dado_real(pasta, imagens) -> None:
    if imagens and not (pasta / MARCADOR_SINTETICO).exists():
        raise ErroProjeto(
            f"{pasta} já tem imagens reais. Não vou misturar dados sintéticos com elas. "
            "Mova suas fotos para outro lugar antes de gerar o sintético."
        )


def _limpar_pasta_sintetica(pasta) -> None:
    if not pasta.exists():
        return
    if listar_imagens(pasta / "images") + listar_imagens(pasta) and not (pasta / MARCADOR_SINTETICO).exists():
        raise ErroProjeto(f"{pasta} tem imagens reais; não vou apagar nada.")
    for item in pasta.iterdir():
        if item.name == ".gitkeep":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()
