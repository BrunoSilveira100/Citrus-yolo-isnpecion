from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from citrus_yolo.config import Config
from citrus_yolo.dataset import Caixa, escrever_rotulo
from citrus_yolo.render import salvar_imagem
from citrus_yolo.sintetico import cena

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def projeto(tmp_path: Path) -> Config:
    """Projeto temporário com a config real do repositório."""
    (tmp_path / "configs").mkdir()
    shutil.copy(RAIZ / "configs" / "projeto.yaml", tmp_path / "configs" / "projeto.yaml")
    shutil.copy(RAIZ / "configs" / "relatorio.yaml", tmp_path / "configs" / "relatorio.yaml")
    return Config.load(tmp_path / "configs" / "projeto.yaml")


def criar_bruto(cfg: Config, n: int, semente: int = 1) -> None:
    import numpy as np

    rng = np.random.default_rng(semente)
    for i in range(n):
        img, caixas = cena(rng, 160, 120)
        if not caixas:
            caixas = [Caixa(i % 2, 0.5, 0.5, 0.2, 0.2)]
        salvar_imagem(cfg.p("bruto") / "images" / f"f{i:03d}.jpg", img)
        escrever_rotulo(cfg.p("bruto") / "labels" / f"f{i:03d}.txt", caixas)
