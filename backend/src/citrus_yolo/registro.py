"""Registro dos comandos realmente executados (item E da entrega).

Cada etapa anexa o comando da CLI e o equivalente na CLI `yolo` do Ultralytics em
resultados/comandos_executados.txt. Assim o comandos.txt da entrega mostra o que foi rodado de
verdade, com data e hora, e não uma lista escrita de memória.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from .config import Config

NOME_ARQUIVO = "comandos_executados.txt"


def linha_de_comando(argv: list[str] | None = None) -> str:
    argumentos = sys.argv[1:] if argv is None else argv
    partes = [f'"{a}"' if (" " in a or not a) else a for a in argumentos]
    return " ".join(["citrus-yolo", *partes])


def registrar(cfg: Config, etapa: str, comando: str, equivalente_yolo: str | None = None) -> Path:
    arquivo = cfg.p("resultados") / NOME_ARQUIVO
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    bloco = [f"# {datetime.now():%Y-%m-%d %H:%M:%S} | {etapa}", comando]
    if equivalente_yolo:
        bloco += ["# equivalente na CLI do Ultralytics:", equivalente_yolo]
    with arquivo.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(bloco) + "\n\n")
    return arquivo
