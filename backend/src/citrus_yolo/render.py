"""Desenho: caixas, painel de inspeção e mosaicos.

Leitura e escrita passam por imdecode/imencode porque o cv2.imread/imwrite do Windows não abre
caminhos com acento (ex.: C:/Users/joão/...). Com np.fromfile/tofile funciona em qualquer caminho.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ErroProjeto

# BGR. Classe 0 verde, classe 1 vermelha; as demais seguem a paleta.
CORES = [(80, 200, 80), (60, 60, 235), (235, 170, 40), (200, 80, 200), (40, 200, 220), (160, 160, 160)]
FONTE = cv2.FONT_HERSHEY_SIMPLEX
FUNDO_PAINEL = (24, 22, 20)
TEXTO_CLARO = (235, 235, 235)
TEXTO_APAGADO = (150, 145, 135)


@dataclass(frozen=True)
class CaixaDesenho:
    xyxy: tuple[float, float, float, float]
    classe: int
    nome: str
    confianca: float | None = None


def cor_classe(indice: int) -> tuple[int, int, int]:
    return CORES[indice % len(CORES)]


def ascii(texto: str) -> str:
    """putText do OpenCV não desenha acento; "INSPEÇÃO" vira "INSPECAO"."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def ler_imagem(caminho: str | Path) -> np.ndarray:
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None
    if imagem is None:
        raise ErroProjeto(f"Não consegui abrir a imagem {caminho} (arquivo corrompido ou formato estranho).")
    return imagem


def salvar_imagem(caminho: str | Path, imagem: np.ndarray, qualidade: int = 92) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    extensao = caminho.suffix.lower() or ".jpg"
    parametros = [cv2.IMWRITE_JPEG_QUALITY, qualidade] if extensao in {".jpg", ".jpeg"} else []
    ok, buffer = cv2.imencode(extensao, imagem, parametros)
    if not ok:
        raise ErroProjeto(f"Não consegui salvar a imagem {caminho}.")
    buffer.tofile(str(caminho))
    return caminho


def desenhar_caixas(imagem: np.ndarray, caixas: Sequence[CaixaDesenho]) -> np.ndarray:
    saida = imagem.copy()
    h, w = saida.shape[:2]
    espessura = max(2, round(min(h, w) / 320))
    escala = max(0.45, min(h, w) / 1100)
    for caixa in caixas:
        cor = cor_classe(caixa.classe)
        x1, y1, x2, y2 = (int(round(v)) for v in caixa.xyxy)
        cv2.rectangle(saida, (x1, y1), (x2, y2), cor, espessura)
        rotulo = ascii(caixa.nome)
        if caixa.confianca is not None:
            rotulo += f" {caixa.confianca:.2f}"
        (tw, th), base = cv2.getTextSize(rotulo, FONTE, escala, 1)
        topo = y1 - th - base - 4
        if topo < 0:  # sem espaço acima da caixa: escreve por dentro
            topo = y1
        cv2.rectangle(saida, (x1, topo), (x1 + tw + 6, topo + th + base + 4), cor, -1)
        cv2.putText(saida, rotulo, (x1 + 3, topo + th + 2), FONTE, escala, (255, 255, 255), 1, cv2.LINE_AA)
    return saida


def painel_inspecao(
    imagem: np.ndarray,
    contagem: dict[str, int],
    classe_defeito: str = "",
    titulo: str = "INSPECAO DE LARANJAS",
    rodape: str = "",
    altura: int = 540,
    largura_painel: int = 300,
) -> np.ndarray:
    """Imagem à esquerda + barra lateral com contagem por classe, total e taxa de defeito."""
    escala = altura / imagem.shape[0]
    largura = max(1, round(imagem.shape[1] * escala))
    vista = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_AREA)

    painel = np.full((altura, largura_painel, 3), FUNDO_PAINEL, np.uint8)
    cv2.putText(painel, ascii(titulo), (18, 38), FONTE, 0.62, TEXTO_CLARO, 2, cv2.LINE_AA)
    cv2.line(painel, (18, 52), (largura_painel - 18, 52), (70, 70, 70), 1)

    linhas: list[tuple[str, str, tuple[int, int, int]]] = []
    for indice, (nome, quantidade) in enumerate(contagem.items()):
        linhas.append((nome, str(quantidade), cor_classe(indice)))
    total = sum(contagem.values())
    linhas.append(("total", str(total), TEXTO_CLARO))
    if classe_defeito and classe_defeito in contagem:
        taxa = 100 * contagem[classe_defeito] / total if total else 0.0
        linhas.append(("taxa de defeito", f"{taxa:.1f}%", TEXTO_CLARO))

    passo = min(78, (altura - 110) // max(1, len(linhas)))
    for i, (nome, valor, cor) in enumerate(linhas):
        y = 88 + i * passo
        cv2.putText(painel, ascii(nome).upper(), (18, y), FONTE, 0.46, TEXTO_APAGADO, 1, cv2.LINE_AA)
        cv2.putText(painel, valor, (18, y + 34), FONTE, 1.0, cor, 2, cv2.LINE_AA)
    if rodape:
        cv2.putText(painel, ascii(rodape)[:34], (18, altura - 18), FONTE, 0.42, TEXTO_APAGADO, 1, cv2.LINE_AA)
    return np.hstack([vista, painel])


def mosaico(imagens: Sequence[np.ndarray], colunas: int = 3, largura_celula: int = 420, espaco: int = 8) -> np.ndarray:
    if not imagens:
        raise ErroProjeto("Nenhuma imagem para montar o mosaico.")
    redimensionadas = []
    for img in imagens:
        escala = largura_celula / img.shape[1]
        redimensionadas.append(
            cv2.resize(img, (largura_celula, max(1, round(img.shape[0] * escala))), interpolation=cv2.INTER_AREA)
        )
    altura_celula = max(img.shape[0] for img in redimensionadas)
    colunas = max(1, min(colunas, len(redimensionadas)))
    linhas = -(-len(redimensionadas) // colunas)
    tela = np.full(
        (linhas * altura_celula + (linhas + 1) * espaco, colunas * largura_celula + (colunas + 1) * espaco, 3),
        255,
        np.uint8,
    )
    for i, img in enumerate(redimensionadas):
        lin, col = divmod(i, colunas)
        y = espaco + lin * (altura_celula + espaco)
        x = espaco + col * (largura_celula + espaco)
        tela[y : y + img.shape[0], x : x + img.shape[1]] = img
    return tela
