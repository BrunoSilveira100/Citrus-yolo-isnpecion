import json
import shutil
from pathlib import Path

import pytest
import yaml

from citrus_yolo.config import ErroProjeto
from citrus_yolo.dataset import (
    Amostra,
    Caixa,
    dividir,
    importar_pasta_yolo,
    ler_rotulo,
    origem,
    preparar,
)

from .conftest import criar_bruto


def rotulo(tmp_path: Path, texto: str) -> Path:
    arquivo = tmp_path / "x.txt"
    arquivo.write_text(texto, encoding="utf-8")
    return arquivo


def test_le_rotulo_valido(tmp_path):
    caixas = ler_rotulo(rotulo(tmp_path, "0 0.5 0.5 0.2 0.3\n\n1 0.1 0.1 0.05 0.05\n"), 2)
    assert caixas == [Caixa(0, 0.5, 0.5, 0.2, 0.3), Caixa(1, 0.1, 0.1, 0.05, 0.05)]


@pytest.mark.parametrize(
    "linha, trecho",
    [
        ("0 0.5 0.5 0.2", "esperava 5 valores"),
        ("0 0.1 0.1 0.2 0.1 0.2 0.2", "polígono"),
        ("2 0.5 0.5 0.2 0.2", "fora do intervalo"),
        ("0 320 240 50 50", "normalizadas"),
        ("0 0.5 0.5 0 0.2", "maiores que zero"),
        ("a 0.5 0.5 0.2 0.2", "não é número"),
        ("0.5 0.5 0.5 0.2 0.2", "inteiro"),
    ],
)
def test_rotulo_invalido_diz_arquivo_e_linha(tmp_path, linha, trecho):
    with pytest.raises(ErroProjeto, match=trecho) as exc:
        ler_rotulo(rotulo(tmp_path, linha), 2)
    assert "x.txt, linha 1" in str(exc.value)


def test_origem_agrupa_copias_do_roboflow():
    assert origem("foto1_jpg.rf.abc123.jpg") == origem("foto1_jpg.rf.zzz999.jpg") == "foto1_jpg"
    assert origem("foto2.jpg") == "foto2"


def _amostra(nome: str, classes: list[int]) -> Amostra:
    return Amostra(Path(nome), Path(nome + ".txt"), [Caixa(c, 0.5, 0.5, 0.1, 0.1) for c in classes], nome)


def test_divisao_estratificada_reprodutivel_e_sem_vazamento():
    amostras = [_amostra(f"boa{i}.jpg", [0]) for i in range(40)]
    amostras += [_amostra(f"def{i}.jpg", [1]) for i in range(10)]
    # 3 cópias aumentadas da mesma foto têm de ir juntas
    amostras += [_amostra(f"foto_jpg.rf.{h}.jpg", [1]) for h in ("a", "b", "c")]

    treino, val = dividir(amostras, 0.2, 42)
    treino2, val2 = dividir(amostras, 0.2, 42)
    assert [a.imagem for a in val] == [a.imagem for a in val2]

    assert not {a.imagem for a in treino} & {a.imagem for a in val}
    assert len(treino) + len(val) == len(amostras)
    assert any(1 in a.classes for a in val), "a classe minoritária precisa aparecer na validação"
    lados = {"treino" if a in treino else "val" for a in amostras if ".rf." in a.imagem.name}
    assert len(lados) == 1


def test_preparar_gera_estrutura_e_data_yaml(projeto):
    criar_bruto(projeto, 60)
    resumo = preparar(projeto)

    base = projeto.p("dataset")
    for pasta in ("images/train", "images/val", "labels/train", "labels/val"):
        assert any((base / pasta).iterdir())
    dados = yaml.safe_load(projeto.data_yaml.read_text(encoding="utf-8"))
    assert dados["train"] == "images/train" and dados["val"] == "images/val"
    assert dados["names"] == {0: "laranja_boa", 1: "laranja_defeito"}
    assert Path(dados["path"]).is_absolute()

    assert resumo["treino"]["imagens"] + resumo["validacao"]["imagens"] == 60
    assert 0.15 <= resumo["proporcao_validacao"] <= 0.25
    salvo = json.loads((projeto.p("resultados") / "dataset" / "resumo_dataset.json").read_text(encoding="utf-8"))
    assert salvo["checklist"]["pelo_menos_60_imagens_anotadas"] is True
    assert salvo["checklist"]["pelo_menos_5_ineditas"] is False


def test_preparar_exige_minimo_de_imagens(projeto):
    criar_bruto(projeto, 20)
    with pytest.raises(ErroProjeto, match="pelo menos 60"):
        preparar(projeto)
    assert preparar(projeto, forcar=True)["total"]["imagens"] == 20


def test_imagem_sem_rotulo_para_o_preparo(projeto):
    criar_bruto(projeto, 60)
    (projeto.p("bruto") / "labels" / "f000.txt").unlink()
    with pytest.raises(ErroProjeto, match="sem o .txt"):
        preparar(projeto)


def test_duplicata_no_bruto_e_ignorada(projeto):
    criar_bruto(projeto, 61)
    bruto = projeto.p("bruto")
    shutil.copy(bruto / "images" / "f000.jpg", bruto / "images" / "z_copia.jpg")
    shutil.copy(bruto / "labels" / "f000.txt", bruto / "labels" / "z_copia.txt")
    resumo = preparar(projeto)
    assert resumo["total"]["imagens"] == 61
    assert any("cópia idêntica" in a for a in resumo["avisos"])


def test_inedita_copiada_do_treino_e_barrada(projeto):
    criar_bruto(projeto, 60)
    projeto.p("ineditas").mkdir(parents=True)
    shutil.copy(projeto.p("bruto") / "images" / "f005.jpg", projeto.p("ineditas") / "nova.jpg")
    with pytest.raises(ErroProjeto, match="nova.jpg"):
        preparar(projeto)


def test_importar_export_roboflow_remapeia_classes_pelo_nome(projeto, tmp_path):
    export = tmp_path / "export"
    for split in ("train", "valid"):
        (export / split / "images").mkdir(parents=True)
        (export / split / "labels").mkdir(parents=True)
        (export / split / "images" / f"{split}1.jpg").write_bytes(b"jpg")
        # no export, 0 = defeito e 1 = boa (ordem invertida em relação ao projeto)
        (export / split / "labels" / f"{split}1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (export / "data.yaml").write_text("names: [laranja_defeito, laranja_boa]\n", encoding="utf-8")

    info = importar_pasta_yolo(projeto, export)
    assert info["copiadas"] == 2
    caixas = ler_rotulo(projeto.p("bruto") / "labels" / "train1.txt", 2)
    assert caixas[0].classe == 1  # laranja_defeito no projeto


def test_importar_rejeita_classe_desconhecida(projeto, tmp_path):
    (tmp_path / "e" / "labels").mkdir(parents=True)
    (tmp_path / "e" / "data.yaml").write_text("names: [limao, laranja_boa]\n", encoding="utf-8")
    with pytest.raises(ErroProjeto, match="limao"):
        importar_pasta_yolo(projeto, tmp_path / "e")
