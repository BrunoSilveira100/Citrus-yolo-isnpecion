"""Leitura e validação do configs/projeto.yaml.

Cada seção do YAML vira uma dataclass, chaves
desconhecidas são rejeitadas (pega erro de digitação cedo) e tudo é validado antes de qualquer etapa
gastar tempo de treino.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ARQUIVO_PADRAO = Path("configs") / "projeto.yaml"
NOME_CLASSE = re.compile(r"^[A-Za-z0-9_]+$")
DISPOSITIVO = re.compile(r"^(cpu|mps|\d+(,\d+)*)$")
EPOCAS_MINIMAS = 20


class ErroProjeto(Exception):
    """Erro esperado (config errada, etapa fora de ordem, arquivo faltando).

    A CLI mostra só a mensagem, sem traceback, porque ela já diz o que fazer.
    """


@dataclass
class Aluno:
    nome: str = ""
    curso: str = ""
    instituicao: str = ""
    disciplina: str = "Inteligência Artificial"
    professor: str = ""


@dataclass
class Projeto:
    titulo: str = "Inspeção de laranjas com YOLO"
    descricao: str = ""


@dataclass
class Caminhos:
    bruto: str = "dataset/bruto"
    dataset: str = "dataset"
    ineditas: str = "dataset/ineditas"
    modelo: str = "modelo"
    resultados: str = "resultados"
    runs: str = "runs/detect"
    entrega: str = "entrega"


@dataclass
class Divisao:
    proporcao_val: float = 0.2
    semente: int = 42
    minimo_imagens: int = 60
    minimo_ineditas: int = 5


@dataclass
class Treino:
    modelo_base: str = "yolov8n.pt"
    epocas: int = 30
    imgsz: int = 640
    batch: int = 4
    dispositivo: str = "cpu"
    paciencia: int = 50
    nome_execucao: str = "train"
    justificativa_epocas: str = ""


@dataclass
class Deteccao:
    confianca: float = 0.25
    iou: float = 0.7
    classe_defeito: str = "laranja_defeito"


@dataclass
class Config:
    aluno: Aluno = field(default_factory=Aluno)
    projeto: Projeto = field(default_factory=Projeto)
    classes: list[str] = field(default_factory=lambda: ["laranja_boa", "laranja_defeito"])
    caminhos: Caminhos = field(default_factory=Caminhos)
    divisao: Divisao = field(default_factory=Divisao)
    treino: Treino = field(default_factory=Treino)
    deteccao: Deteccao = field(default_factory=Deteccao)
    # raiz não vem do YAML: é a pasta do projeto, usada para resolver caminhos relativos
    raiz: Path = field(default_factory=Path.cwd, init=False, repr=False)

    @classmethod
    def load(cls, caminho: str | Path) -> Config:
        caminho = Path(caminho).resolve()
        if not caminho.is_file():
            raise ErroProjeto(f"Arquivo de configuração não encontrado: {caminho}")
        try:
            bruto = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ErroProjeto(f"YAML inválido em {caminho}: {exc}") from exc
        if not isinstance(bruto, dict):
            raise ErroProjeto(f"O conteúdo de {caminho} precisa ser um mapa (chave: valor).")
        cfg = _montar(cls, bruto, "")
        cfg.raiz = caminho.parent.parent if caminho.parent.name == "configs" else caminho.parent
        cfg.validar()
        return cfg

    # ---------- caminhos ----------
    def p(self, nome: str) -> Path:
        """Caminho absoluto de uma entrada da seção `caminhos` (ex.: cfg.p("bruto"))."""
        valor = Path(getattr(self.caminhos, nome))
        return valor if valor.is_absolute() else (self.raiz / valor).resolve()

    @property
    def data_yaml(self) -> Path:
        return self.p("dataset") / "data.yaml"

    @property
    def best_pt(self) -> Path:
        return self.p("modelo") / "best.pt"

    def relativo(self, caminho: Path) -> str:
        """Caminho relativo à raiz com "/" (fica igual no Windows e no Linux, bom para logs)."""
        try:
            return Path(caminho).resolve().relative_to(self.raiz).as_posix()
        except ValueError:
            return Path(caminho).as_posix()

    # ---------- validação ----------
    def validar(self) -> None:
        if not isinstance(self.classes, list) or len(self.classes) < 2:
            raise ErroProjeto("A atividade exige pelo menos 2 classes em `classes`.")
        for nome in self.classes:
            if not isinstance(nome, str) or not NOME_CLASSE.match(nome):
                raise ErroProjeto(
                    f"Nome de classe inválido: {nome!r}. Use só letras, números e '_' (ex.: laranja_boa)."
                )
        if len(set(self.classes)) != len(self.classes):
            raise ErroProjeto("Há nomes de classe repetidos em `classes`.")

        d = self.divisao
        _numero("divisao.proporcao_val", d.proporcao_val)
        if not 0 < d.proporcao_val <= 0.5:
            raise ErroProjeto("divisao.proporcao_val deve ficar entre 0 e 0.5 (o recomendado é 0.2).")
        _inteiro("divisao.semente", d.semente, minimo=0)
        _inteiro("divisao.minimo_imagens", d.minimo_imagens, minimo=1)
        _inteiro("divisao.minimo_ineditas", d.minimo_ineditas, minimo=1)

        t = self.treino
        if not isinstance(t.modelo_base, str) or not t.modelo_base.endswith((".pt", ".yaml")):
            raise ErroProjeto("treino.modelo_base deve ser um arquivo .pt ou .yaml (ex.: yolov8n.pt).")
        _inteiro("treino.epocas", t.epocas, minimo=1)
        if t.epocas < EPOCAS_MINIMAS and not str(t.justificativa_epocas).strip():
            raise ErroProjeto(
                f"A atividade pede pelo menos {EPOCAS_MINIMAS} épocas. Para usar {t.epocas}, "
                "explique o motivo em treino.justificativa_epocas (ou use --justificativa na CLI)."
            )
        _inteiro("treino.imgsz", t.imgsz, minimo=32)
        if t.imgsz % 32:
            raise ErroProjeto(f"treino.imgsz precisa ser múltiplo de 32 (ex.: 320, 640). Valor: {t.imgsz}.")
        _inteiro("treino.batch", t.batch, minimo=-1)
        if t.batch == 0:
            raise ErroProjeto("treino.batch não pode ser 0 (use um inteiro positivo, ou -1 para automático).")
        t.dispositivo = str(t.dispositivo).strip().lower()
        if not DISPOSITIVO.match(t.dispositivo):
            raise ErroProjeto(f"treino.dispositivo inválido: {t.dispositivo!r}. Use 'cpu' ou '0' (GPU).")
        _inteiro("treino.paciencia", t.paciencia, minimo=0)
        if not NOME_CLASSE.match(str(t.nome_execucao)):
            raise ErroProjeto("treino.nome_execucao deve ter só letras, números e '_'.")

        det = self.deteccao
        for nome in ("confianca", "iou"):
            valor = getattr(det, nome)
            _numero(f"deteccao.{nome}", valor)
            if not 0 < valor <= 1:
                raise ErroProjeto(f"deteccao.{nome} deve estar entre 0 e 1.")
        if det.classe_defeito and det.classe_defeito not in self.classes:
            raise ErroProjeto(
                f"deteccao.classe_defeito ({det.classe_defeito!r}) não está em `classes`. "
                "Ajuste o nome ou deixe vazio para não calcular taxa de defeito."
            )


def localizar_config(informado: Path | None = None) -> Path:
    """Acha o projeto.yaml: o informado em --config ou subindo a partir da pasta atual."""
    if informado is not None:
        if not informado.is_file():
            raise ErroProjeto(f"Config não encontrada: {informado}")
        return informado
    pasta = Path.cwd().resolve()
    for candidata in (pasta, *pasta.parents):
        # funciona tanto de dentro de backend/ quanto da raiz do repositório
        for arquivo in (candidata / ARQUIVO_PADRAO, candidata / "backend" / ARQUIVO_PADRAO):
            if arquivo.is_file():
                return arquivo
    raise ErroProjeto(
        "Não achei backend/configs/projeto.yaml. Rode o comando de dentro do projeto ou use --config."
    )


def _montar(cls: type, bruto: Any, prefixo: str):
    if not isinstance(bruto, dict):
        raise ErroProjeto(f"A seção '{prefixo or cls.__name__}' precisa ser um mapa (chave: valor).")
    conhecidos = {f.name for f in fields(cls) if f.init}
    desconhecidos = set(bruto) - conhecidos
    if desconhecidos:
        onde = f" em '{prefixo}'" if prefixo else ""
        raise ErroProjeto(f"Chave(s) desconhecida(s){onde}: {sorted(desconhecidos)}. Confira a digitação.")
    padrao = cls()
    valores = {}
    for nome, valor in bruto.items():
        atual = getattr(padrao, nome)
        caminho = f"{prefixo}.{nome}" if prefixo else nome
        valores[nome] = _montar(type(atual), valor, caminho) if is_dataclass(atual) else valor
    return cls(**valores)


def _numero(nome: str, valor: Any) -> None:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErroProjeto(f"{nome} precisa ser um número.")


def _inteiro(nome: str, valor: Any, minimo: int) -> None:
    if isinstance(valor, bool) or not isinstance(valor, int) or valor < minimo:
        raise ErroProjeto(f"{nome} precisa ser um inteiro >= {minimo}.")
