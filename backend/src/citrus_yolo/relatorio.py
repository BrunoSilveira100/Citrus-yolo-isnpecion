"""Gera o relatório (Markdown + PDF) a partir dos arquivos de resultados/.

Regra de ouro: nenhum número é digitado aqui. Tudo vem de resumo_dataset.json, treino.json,
metricas.json e resumo_deteccoes.json. Se um arquivo não existe, a seção mostra "PENDENTE" e o
comando que falta rodar. Os textos pessoais vêm de configs/relatorio.yaml.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import yaml

from .config import Config, ErroProjeto
from .dataset import MARCADOR_SINTETICO

TEXTOS_PADRAO = {
    "questao1": "",
    "obtencao_imagens": "",
    "link_dataset": "",
    "explicacao_metricas": {},
    "comentario_ineditas": "",
    "dificuldades": "",
    "melhorias": "",
}


@dataclass
class Dados:
    dataset: dict | None
    treino: dict | None
    metricas: dict | None
    deteccoes: dict | None
    textos: dict
    sintetico: bool
    imagens: dict[str, Path] = field(default_factory=dict)


def _json(caminho: Path) -> dict | None:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.is_file() else None


def carregar_textos(cfg: Config) -> dict:
    arquivo = cfg.raiz / "configs" / "relatorio.yaml"
    if not arquivo.is_file():
        return dict(TEXTOS_PADRAO)
    try:
        dados = yaml.safe_load(arquivo.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ErroProjeto(f"YAML inválido em configs/relatorio.yaml: {exc}") from exc
    return {**TEXTOS_PADRAO, **dados}


def coletar(cfg: Config) -> Dados:
    r = cfg.p("resultados")
    candidatas = {
        "exemplos": r / "dataset" / "exemplos_anotados.jpg",
        "results": r / "treino" / "results.png",
        "confusao": r / "validacao" / "confusion_matrix.png",
        "pr": next(iter(sorted((r / "validacao").glob("*PR_curve.png"))), r / "_inexistente"),
        "mosaico": r / "deteccoes" / "mosaico_deteccoes.jpg",
    }
    return Dados(
        dataset=_json(r / "dataset" / "resumo_dataset.json"),
        treino=_json(r / "treino" / "treino.json"),
        metricas=_json(r / "validacao" / "metricas.json"),
        deteccoes=_json(r / "deteccoes" / "resumo_deteccoes.json"),
        textos=carregar_textos(cfg),
        sintetico=(cfg.p("bruto") / MARCADOR_SINTETICO).exists(),
        imagens={k: v for k, v in candidatas.items() if v.is_file()},
    )


def pontos_de_atencao(metricas: dict | None, deteccoes: dict | None) -> list[str]:
    """Leituras automáticas dos números, para puxar a análise crítica da questão 6."""
    pontos = []
    if metricas:
        por_classe = metricas.get("por_classe", {})
        if len(por_classe) >= 2:
            pior = min(por_classe, key=lambda c: por_classe[c]["mAP50"])
            pontos.append(
                f"A classe com pior mAP50 é '{pior}' ({por_classe[pior]['mAP50']:.3f}); é onde mais vale "
                "investir em fotos novas."
            )
        g = metricas["geral"]
        if g["precision"] - g["recall"] > 0.1:
            pontos.append(
                f"Precision ({g['precision']:.2f}) bem acima do recall ({g['recall']:.2f}): o modelo é "
                "conservador. Quando marca, costuma acertar, mas deixa laranjas passarem."
            )
        elif g["recall"] - g["precision"] > 0.1:
            pontos.append(
                f"Recall ({g['recall']:.2f}) bem acima da precision ({g['precision']:.2f}): o modelo acha "
                "quase tudo, mas gera alarmes falsos."
            )
        if g["mAP50"] - g["mAP50_95"] > 0.3:
            pontos.append(
                "Diferença grande entre mAP50 e mAP50-95: o modelo acha as frutas, mas as caixas ainda "
                "não ficam bem ajustadas às bordas."
            )
        faltando = metricas.get("classes_sem_instancia_na_validacao") or []
        if faltando:
            pontos.append(f"Sem exemplos na validação para: {', '.join(faltando)}; a métrica dessas classes não existe.")
    if deteccoes:
        baixas = [
            (img["arquivo"], d["classe"], d["confianca"])
            for img in deteccoes["imagens"]
            for d in img["deteccoes"]
            if d["confianca"] < 0.5
        ]
        if baixas:
            exemplo = baixas[0]
            pontos.append(
                f"{len(baixas)} detecção(ões) com confiança abaixo de 0,5 nas inéditas "
                f"(ex.: {exemplo[1]} {exemplo[2]:.2f} em {exemplo[0]}). São casos em que o modelo está em dúvida."
            )
        if deteccoes.get("imagens_sem_deteccao"):
            pontos.append(f"Nenhuma laranja detectada em: {', '.join(deteccoes['imagens_sem_deteccao'])}.")
        avaliadas = [img["avaliacao"] for img in deteccoes["imagens"] if img.get("avaliacao")]
        if avaliadas:
            erros = sum(1 for a in avaliadas if a["situacao"] == "erro")
            pontos.append(f"Comparando com o gabarito, {len(avaliadas) - erros} de {len(avaliadas)} imagens inéditas sem nenhum erro.")
    return pontos


def pendencias_textos(textos: dict) -> list[str]:
    pendentes = []
    for chave, valor in textos.items():
        valores = valor.values() if isinstance(valor, dict) else [valor]
        if any(isinstance(v, str) and "EDITE:" in v for v in valores):
            pendentes.append(chave)
    return pendentes


# --------------------------------------------------------------------------- conteúdo (estrutura neutra)
# Cada seção é uma lista de blocos ("p", texto) | ("tabela", linhas) | ("img", chave, legenda) | ("alerta", texto).
# O mesmo conteúdo vira Markdown e PDF, então os dois nunca ficam diferentes.


def _f(valor: float) -> str:
    return f"{valor:.3f}"


def secoes(cfg: Config, d: Dados) -> list[tuple[str, list[tuple]]]:
    t = d.textos
    pend = "PENDENTE: rode {cmd}"
    saida: list[tuple[str, list[tuple]]] = []

    saida.append(("Descrição do projeto", [("p", cfg.projeto.descricao or "PENDENTE: preencha projeto.descricao")]))

    saida.append(("Questão 1. Qual problema foi escolhido e por quê?", [("p", t.get("questao1", ""))]))

    q2: list[tuple] = []
    if d.dataset:
        tot, tr, va = d.dataset["total"], d.dataset["treino"], d.dataset["validacao"]
        q2.append(
            (
                "p",
                f"Foram usadas {tot['imagens']} imagens ({tot['imagens_anotadas']} com anotação): "
                f"{tr['imagens']} no treino e {va['imagens']} na validação "
                f"({d.dataset['proporcao_validacao'] * 100:.0f}% para validação, semente {d.dataset['semente']}). "
                f"Outras {d.dataset['ineditas']['quantidade']} imagens ficaram separadas como inéditas.",
            )
        )
        linhas = [["Classe", "Caixas treino", "Caixas val", "Imagens com a classe"]]
        for i, nome in enumerate(d.dataset["classes"]):
            linhas.append([f"{i}: {nome}", str(tr["instancias"][nome]), str(va["instancias"][nome]),
                           str(tot["imagens_por_classe"][nome])])
        q2.append(("tabela", linhas))
    else:
        q2.append(("alerta", pend.format(cmd="citrus-yolo preparar")))
    q2.append(("p", t.get("obtencao_imagens", "")))
    if t.get("link_dataset"):
        q2.append(("p", f"Link do dataset: {t['link_dataset']}"))
    q2.append(("img", "exemplos", "Exemplos de imagens do treino com as caixas anotadas."))
    saida.append(("Questão 2. Quantas imagens, quais classes e como foram obtidas?", q2))

    q3: list[tuple] = []
    if d.treino:
        tr = d.treino
        q3.append(
            (
                "p",
                f"Modelo base {tr['modelo_base']} (YOLOv8 nano, pesos pré-treinados no COCO), ajustado com o "
                f"dataset do projeto. Épocas pedidas: {tr['epocas_pedidas']}; executadas: {tr['epocas_executadas']}. "
                f"imgsz={tr['imgsz']}, batch={tr['batch']}, dispositivo={tr['dispositivo']}, "
                f"tempo total {tr['duracao_s'] / 60:.1f} min. Melhores pesos salvos em {tr['best_pt']} "
                f"({tr['best_pt_mb']} MB), copiados de {tr['best_pt_original']}.",
            )
        )
        if tr.get("justificativa_epocas") and tr["epocas_pedidas"] < 20:
            q3.append(("p", f"Justificativa para menos de 20 épocas: {tr['justificativa_epocas']}"))
        if tr.get("melhor_epoca"):
            m = tr["melhor_epoca"]
            q3.append(("p", f"Época com maior mAP50-95 na validação: {m.get('epoca')} (mAP50 {m.get('mAP50', 0):.3f})."))
    else:
        q3.append(("alerta", pend.format(cmd="citrus-yolo treinar")))
    q3.append(("img", "results", "Curvas do treino (losses caindo e métricas subindo a cada época)."))
    saida.append(("Questão 3. Qual modelo YOLO e quantas épocas?", q3))

    q4: list[tuple] = []
    if d.metricas:
        linhas = [["Conjunto", "Precision", "Recall", "mAP50", "mAP50-95"]]
        g = d.metricas["geral"]
        linhas.append(["todas", _f(g["precision"]), _f(g["recall"]), _f(g["mAP50"]), _f(g["mAP50_95"])])
        for nome, v in d.metricas["por_classe"].items():
            linhas.append([nome, _f(v["precision"]), _f(v["recall"]), _f(v["mAP50"]), _f(v["mAP50_95"])])
        q4.append(("tabela", linhas))
    else:
        q4.append(("alerta", pend.format(cmd="citrus-yolo validar")))
    explicacoes = t.get("explicacao_metricas") or {}
    for chave, rotulo in (("precision", "Precision"), ("recall", "Recall"), ("mAP50", "mAP50"), ("mAP50_95", "mAP50-95")):
        if explicacoes.get(chave):
            q4.append(("p", f"{rotulo}: {explicacoes[chave].strip()}"))
    q4.append(("img", "confusao", "Matriz de confusão na validação."))
    saida.append(("Questão 4. Precision, Recall e mAP50 obtidos e o que significam", q4))

    q5: list[tuple] = []
    if d.deteccoes:
        dd = d.deteccoes
        q5.append(
            (
                "p",
                f"O modelo foi aplicado em {dd['total_imagens']} imagens inéditas com confiança mínima "
                f"{dd['confianca_minima']} e fez {dd['total_deteccoes']} detecções.",
            )
        )
        cab = ["Imagem", "Detecções (classe e confiança)"]
        tem_gabarito = dd.get("com_gabarito", 0) > 0
        if tem_gabarito:
            cab.append("Resultado")
        linhas = [cab]
        for img in dd["imagens"]:
            dets = ", ".join(f"{x['classe']} {x['confianca']:.2f}" for x in img["deteccoes"]) or "nenhuma"
            linha = [img["arquivo"], dets]
            if tem_gabarito:
                a = img.get("avaliacao")
                linha.append(
                    "-" if not a else f"{a['situacao']} (ok {a['acerto']}, troca {a['classe_trocada']}, "
                    f"FP {a['falso_positivo']}, FN {a['nao_detectada']})"
                )
            linhas.append(linha)
        q5.append(("tabela", linhas))
    else:
        q5.append(("alerta", pend.format(cmd="citrus-yolo detectar")))
    q5.append(("p", t.get("comentario_ineditas", "")))
    q5.append(("img", "mosaico", "Painéis de inspeção gerados nas imagens inéditas."))
    saida.append(("Questão 5. O modelo reconheceu os objetos nas imagens inéditas?", q5))

    q6: list[tuple] = [("p", t.get("dificuldades", "")), ("p", "O que melhorar: " + str(t.get("melhorias", "")))]
    pontos = pontos_de_atencao(d.metricas, d.deteccoes)
    if pontos:
        q6.append(("p", "Pontos de atenção tirados dos próprios resultados:"))
        q6 += [("p", f"- {p}") for p in pontos]
    saida.append(("Questão 6. Dificuldades e o que poderia melhorar", q6))
    return saida


# --------------------------------------------------------------------------- Markdown


def gerar_markdown(cfg: Config, d: Dados, pasta_md: Path) -> str:
    a = cfg.aluno
    md = [f"# {cfg.projeto.titulo}", ""]
    if d.sintetico:
        md += ["> **ATENÇÃO: dataset SINTÉTICO, só para teste do pipeline. Não entregar.**", ""]
    md += [
        f"**Aluno:** {a.nome}  ",
        f"**Curso:** {a.curso}  ",
        f"**Instituição:** {a.instituicao}  ",
        f"**Disciplina:** {a.disciplina}  ",
        f"**Professor:** {a.professor}  ",
        f"**Data:** {date.today():%d/%m/%Y}",
        "",
    ]
    for titulo, blocos in secoes(cfg, d):
        md += [f"## {titulo}", ""]
        for bloco in blocos:
            tipo = bloco[0]
            if tipo == "p" and str(bloco[1]).strip():
                md += [str(bloco[1]).strip(), ""]
            elif tipo == "alerta":
                md += [f"> **{bloco[1]}**", ""]
            elif tipo == "tabela":
                linhas = bloco[1]
                md.append("| " + " | ".join(linhas[0]) + " |")
                md.append("|" + " --- |" * len(linhas[0]))
                md += ["| " + " | ".join(linha) + " |" for linha in linhas[1:]]
                md.append("")
            elif tipo == "img" and bloco[1] in d.imagens:
                rel = Path(_relpath(d.imagens[bloco[1]], pasta_md)).as_posix()
                md += [f"![{bloco[2]}]({rel})", "", f"*{bloco[2]}*", ""]
    return "\n".join(md)


def _relpath(alvo: Path, base: Path) -> str:
    import os

    return os.path.relpath(alvo, base)


# --------------------------------------------------------------------------- PDF


def gerar_pdf(cfg: Config, d: Dados, destino: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    estilos = getSampleStyleSheet()
    corpo = ParagraphStyle("corpo", parent=estilos["BodyText"], fontSize=10.5, leading=14.5, spaceAfter=6)
    titulo = ParagraphStyle("titulo", parent=estilos["Title"], fontSize=22, leading=28, spaceAfter=18)
    h2 = ParagraphStyle("h2", parent=estilos["Heading2"], textColor=colors.HexColor("#c2410c"), spaceBefore=12)
    legenda = ParagraphStyle("legenda", parent=corpo, fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER)
    alerta = ParagraphStyle("alerta", parent=corpo, textColor=colors.HexColor("#b91c1c"))
    capa = ParagraphStyle("capa", parent=corpo, alignment=TA_CENTER, fontSize=12, leading=18)
    celula = ParagraphStyle("celula", parent=corpo, fontSize=8.5, leading=10.5, spaceAfter=0)

    largura_util = A4[0] - 4 * cm
    historia: list = [Spacer(1, 4 * cm), Paragraph(escape(cfg.projeto.titulo), titulo)]
    historia.append(Paragraph("Atividade Prática Final: treinamento de um modelo YOLO", capa))
    historia.append(Spacer(1, 2 * cm))
    a = cfg.aluno
    for rotulo, valor in (
        ("Aluno", a.nome),
        ("Curso", a.curso),
        ("Instituição", a.instituicao),
        ("Disciplina", a.disciplina),
        ("Professor", a.professor),
        ("Data", f"{date.today():%d/%m/%Y}"),
    ):
        historia.append(Paragraph(f"<b>{rotulo}:</b> {escape(valor or '-')}", capa))
    if d.sintetico:
        historia += [Spacer(1, 1.5 * cm), Paragraph("<b>ATENÇÃO: dataset SINTÉTICO, só para teste. Não entregar.</b>", alerta)]
    historia.append(PageBreak())

    for titulo_secao, blocos in secoes(cfg, d):
        historia.append(Paragraph(escape(titulo_secao), h2))
        for bloco in blocos:
            tipo = bloco[0]
            if tipo == "p" and str(bloco[1]).strip():
                historia.append(Paragraph(escape(str(bloco[1]).strip()).replace("\n", " "), corpo))
            elif tipo == "alerta":
                historia.append(Paragraph(f"<b>{escape(bloco[1])}</b>", alerta))
            elif tipo == "tabela":
                dados = [[Paragraph(escape(c), celula) for c in linha] for linha in bloco[1]]
                tabela = Table(dados, repeatRows=1, colWidths=[largura_util / len(bloco[1][0])] * len(bloco[1][0]))
                tabela.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fed7aa")),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6d3d1")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )
                historia += [tabela, Spacer(1, 6)]
            elif tipo == "img" and bloco[1] in d.imagens:
                historia += [_imagem(Image, d.imagens[bloco[1]], largura_util, 11 * cm), Paragraph(escape(bloco[2]), legenda)]

    prints = sorted(p for p in (cfg.p("resultados") / "prints").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if prints:
        historia += [PageBreak(), Paragraph("Anexo: capturas de tela do treinamento", h2)]
        for p in prints:
            historia += [_imagem(Image, p, largura_util, 10 * cm), Paragraph(escape(p.name), legenda)]

    destino.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(destino), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title=cfg.projeto.titulo, author=a.nome)
    doc.build(historia)
    return destino


def _imagem(Image, caminho: Path, largura_max: float, altura_max: float):
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(str(caminho)).getSize()
    escala = min(largura_max / w, altura_max / h)
    return Image(str(caminho), width=w * escala, height=h * escala)


def gerar(cfg: Config) -> dict:
    d = coletar(cfg)
    pasta = cfg.p("resultados") / "relatorio"
    pasta.mkdir(parents=True, exist_ok=True)
    md = pasta / "relatorio.md"
    md.write_text(gerar_markdown(cfg, d, pasta), encoding="utf-8")
    pdf = gerar_pdf(cfg, d, pasta / "relatorio.pdf")
    faltando = [nome for nome, valor in (("dataset", d.dataset), ("treino", d.treino),
                                         ("validacao", d.metricas), ("deteccoes", d.deteccoes)) if not valor]
    return {
        "markdown": cfg.relativo(md),
        "pdf": cfg.relativo(pdf),
        "etapas_faltando": faltando,
        "textos_para_editar": pendencias_textos(d.textos),
        "sintetico": d.sintetico,
    }
