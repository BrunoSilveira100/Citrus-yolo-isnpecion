"""Render, métricas, comparação com gabarito, relatório e ZIP — tudo sem torch."""

import json
import zipfile

import numpy as np

from citrus_yolo.deteccao import comparar, iou
from citrus_yolo.entrega import entregar, slug
from citrus_yolo.relatorio import gerar, pendencias_textos, pontos_de_atencao
from citrus_yolo.render import CaixaDesenho, ler_imagem, mosaico, painel_inspecao, salvar_imagem
from citrus_yolo.treino import formatar_epocas, metricas_markdown, montar_metricas, tabela_epocas

from .conftest import criar_bruto


def test_painel_tem_barra_lateral_e_aceita_caminho_com_acento(tmp_path):
    img = np.zeros((240, 320, 3), np.uint8)
    painel = painel_inspecao(img, {"laranja_boa": 3, "laranja_defeito": 1}, "laranja_defeito", altura=540)
    assert painel.shape == (540, 720 + 300, 3)
    destino = salvar_imagem(tmp_path / "inspeção" / "painél.jpg", painel)
    assert ler_imagem(destino).shape == painel.shape


def test_mosaico_organiza_em_grade():
    imgs = [np.zeros((100, 200, 3), np.uint8)] * 5
    assert mosaico(imgs, colunas=3, largura_celula=200, espaco=0).shape == (200, 600, 3)


def test_iou_e_comparacao_com_gabarito():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    gabarito = [CaixaDesenho((0, 0, 10, 10), 0, "boa"), CaixaDesenho((50, 50, 60, 60), 1, "def"),
                CaixaDesenho((80, 80, 90, 90), 0, "boa")]
    previstas = [CaixaDesenho((0, 0, 10, 11), 0, "boa", 0.9), CaixaDesenho((50, 50, 60, 60), 0, "boa", 0.8),
                 CaixaDesenho((200, 200, 210, 210), 1, "def", 0.4)]
    r = comparar(gabarito, previstas)
    assert r == {"acerto": 1, "classe_trocada": 1, "falso_positivo": 1, "nao_detectada": 1, "situacao": "erro"}


def test_tabela_de_epocas_le_csv_do_ultralytics(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text(
        "                  epoch,      train/box_loss,      train/cls_loss,      train/dfl_loss,"
        "   metrics/precision(B),      metrics/recall(B),       metrics/mAP50(B),    metrics/mAP50-95(B)\n"
        "1,1.5,2.5,1.2,0.3,0.4,0.25,0.1\n2,1.2,1.9,1.1,0.6,0.5,0.55,0.3\n",
        encoding="utf-8",
    )
    linhas = tabela_epocas(csv)
    assert linhas[1] == {"epoca": 2, "box_loss": 1.2, "cls_loss": 1.9, "dfl_loss": 1.1,
                         "precision": 0.6, "recall": 0.5, "mAP50": 0.55, "mAP50_95": 0.3}
    assert "mAP50" in formatar_epocas(linhas).splitlines()[0]


def test_metricas_markdown_e_pontos_de_atencao():
    m = montar_metricas(
        {"precision": 0.9, "recall": 0.6, "mAP50": 0.8, "mAP50_95": 0.4},
        {"laranja_boa": {"precision": 0.95, "recall": 0.7, "mAP50": 0.9, "mAP50_95": 0.5},
         "laranja_defeito": {"precision": 0.85, "recall": 0.5, "mAP50": 0.7, "mAP50_95": 0.3}},
        ["laranja_boa", "laranja_defeito"],
        "modelo/best.pt",
    )
    md = metricas_markdown(m)
    assert "| laranja_defeito | 0.850 | 0.500 | 0.700 | 0.300 |" in md
    pontos = " ".join(pontos_de_atencao(m, None))
    assert "laranja_defeito" in pontos and "conservador" in pontos and "mAP50-95" in pontos


def test_relatorio_sem_resultados_marca_pendente_e_nao_inventa_numero(projeto):
    (projeto.raiz / "configs" / "relatorio.yaml").write_text("questao1: 'EDITE: rascunho'\n", encoding="utf-8")
    info = gerar(projeto)
    md = (projeto.raiz / info["markdown"]).read_text(encoding="utf-8")
    assert "PENDENTE: rode citrus-yolo treinar" in md
    assert "PENDENTE: rode citrus-yolo validar" in md
    assert (projeto.raiz / info["pdf"]).stat().st_size > 1000
    assert set(info["etapas_faltando"]) == {"dataset", "treino", "validacao", "deteccoes"}
    assert "questao1" in info["textos_para_editar"]


def test_pendencias_de_texto():
    assert pendencias_textos({"a": "EDITE: x", "b": "pronto", "c": {"d": "EDITE: y"}}) == ["a", "c"]


def test_zip_segue_estrutura_da_atividade(projeto):
    from citrus_yolo.dataset import preparar

    criar_bruto(projeto, 60)
    preparar(projeto)
    projeto.best_pt.parent.mkdir(parents=True)
    projeto.best_pt.write_bytes(b"pesos-falsos-so-para-teste")
    met = projeto.p("resultados") / "validacao" / "metricas.json"
    met.parent.mkdir(parents=True)
    valores = {"precision": 0.5, "recall": 0.5, "mAP50": 0.5, "mAP50_95": 0.2}
    met.write_text(json.dumps({"geral": valores, "por_classe": {}}), encoding="utf-8")
    gerar(projeto)

    info = entregar(projeto)
    with zipfile.ZipFile(projeto.raiz / info["zip"]) as z:
        nomes = set(z.namelist())
        raiz = f"Projeto_Final_YOLO_{slug(projeto.aluno.nome)}/"
        for esperado in ("dataset/data.yaml", "modelo/best.pt", "relatorio.pdf", "comandos.txt",
                         "resultados/validacao/metricas.json", "codigo/configs/projeto.yaml"):
            assert raiz + esperado in nomes
        assert any(n.startswith(raiz + "dataset/images/train/") for n in nomes)
        assert "path:" not in z.read(raiz + "dataset/data.yaml").decode("utf-8")
    assert any("detecção" in p for p in info["pendencias"])


def test_slug():
    assert slug("João da Silva") == "Joao_da_Silva"
    assert slug("") == "SeuNome"
