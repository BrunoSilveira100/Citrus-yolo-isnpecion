from pathlib import Path

import pytest

from citrus_yolo.config import Config, ErroProjeto


def escrever(tmp_path: Path, texto: str) -> Path:
    (tmp_path / "configs").mkdir(exist_ok=True)
    arquivo = tmp_path / "configs" / "projeto.yaml"
    arquivo.write_text(texto, encoding="utf-8")
    return arquivo


def test_config_do_repositorio_e_valida(projeto):
    assert projeto.classes == ["laranja_boa", "laranja_defeito"]
    assert projeto.treino.epocas >= 20
    assert projeto.data_yaml == projeto.raiz / "dataset" / "data.yaml"


def test_raiz_e_a_pasta_acima_de_configs(tmp_path):
    cfg = Config.load(escrever(tmp_path, "classes: [a, b]\ndeteccao: {classe_defeito: b}\n"))
    assert cfg.raiz == tmp_path.resolve()
    assert cfg.p("bruto") == (tmp_path / "dataset" / "bruto").resolve()


@pytest.mark.parametrize(
    "texto, trecho",
    [
        ("classes: [so_uma]\n", "pelo menos 2 classes"),
        ("classes: [laranja boa, ruim]\n", "Nome de classe inválido"),
        ("classes: [a, a]\n", "repetidos"),
        ("classes: [a, b]\ntreino: {imgsz: 300}\n", "múltiplo de 32"),
        ("classes: [a, b]\ntreino: {epocas: 5}\n", "justificativa"),
        ("classes: [a, b]\ntreino: {dispositivo: gpu}\n", "dispositivo"),
        ("classes: [a, b]\ntreinoo: {}\n", "desconhecida"),
        ("classes: [a, b]\ndeteccao: {classe_defeito: c}\n", "classe_defeito"),
    ],
)
def test_config_invalida(tmp_path, texto, trecho):
    with pytest.raises(ErroProjeto, match=trecho):
        Config.load(escrever(tmp_path, texto))


def test_menos_de_20_epocas_com_justificativa(tmp_path):
    texto = "classes: [a, b]\ndeteccao: {classe_defeito: b}\ntreino: {epocas: 5, justificativa_epocas: teste}\n"
    assert Config.load(escrever(tmp_path, texto)).treino.epocas == 5


def test_dispositivo_numerico_vira_texto(tmp_path):
    cfg = Config.load(escrever(tmp_path, "classes: [a, b]\ndeteccao: {classe_defeito: b}\ntreino: {dispositivo: 0}\n"))
    assert cfg.treino.dispositivo == "0"
