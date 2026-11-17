"""CLI do projeto: `citrus-yolo <etapa>`.

Ordem normal:  preparar → treinar → validar → detectar → relatorio → entregar   (ou `tudo`)
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import dataset as ds
from .config import Config, ErroProjeto, localizar_config
from .registro import linha_de_comando, registrar

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inspeção de laranjas com YOLO: do dataset ao ZIP de entrega.",
)
console = Console()
erro = Console(stderr=True)

OPCAO_CONFIG = typer.Option(None, "--config", "-c", help="Caminho do projeto.yaml (padrão: configs/projeto.yaml).")


@contextmanager
def tratar_erros():
    try:
        yield
    except ErroProjeto as exc:
        erro.print(f"[bold red]Erro:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt as exc:
        erro.print("[yellow]Interrompido.[/yellow]")
        raise typer.Exit(130) from exc


def carregar(config: Path | None) -> Config:
    return Config.load(localizar_config(config))


def ajustar_treino(cfg: Config, epocas=None, justificativa=None, dispositivo=None, imgsz=None, batch=None) -> None:
    if epocas is not None:
        cfg.treino.epocas = epocas
    if justificativa:
        cfg.treino.justificativa_epocas = justificativa
    if dispositivo is not None:
        cfg.treino.dispositivo = dispositivo
    if imgsz is not None:
        cfg.treino.imgsz = imgsz
    if batch is not None:
        cfg.treino.batch = batch
    cfg.validar()


def usar_imgsz_do_treino(cfg: Config, imgsz: int | None) -> None:
    """validar/detectar usam o mesmo imgsz do treino, a menos que o usuário peça outro."""
    if imgsz is not None:
        cfg.treino.imgsz = imgsz
        return
    from .treino import info_treino

    info = info_treino(cfg)
    if info:
        cfg.treino.imgsz = int(info["imgsz"])


def _rel(cfg: Config, caminho: Path) -> str:
    return cfg.relativo(caminho)


def _mostrar_avisos(avisos: list[str]) -> None:
    for aviso in avisos:
        console.print(f"[yellow]Aviso:[/yellow] {aviso}")


# --------------------------------------------------------------------------- dados


@app.command()
def sintetico(
    imagens: int = typer.Option(80, min=10, help="Quantas imagens de treino/val gerar."),
    ineditas: int = typer.Option(6, min=1, help="Quantas imagens inéditas gerar."),
    semente: int = typer.Option(7),
    limpar: bool = typer.Option(False, "--limpar", help="Apaga o sintético e tudo o que foi gerado dele."),
    config: Path | None = OPCAO_CONFIG,
) -> None:
    """Gera laranjas ARTIFICIAIS só para testar o pipeline (não usar na entrega)."""
    with tratar_erros():
        from . import sintetico as sint

        cfg = carregar(config)
        if limpar:
            removidos = sint.limpar(cfg)
            console.print("[green]Removido:[/green] " + ", ".join(removidos))
            return
        info = sint.gerar(cfg, imagens, ineditas, semente)
        console.print(f"[green]Gerado:[/green] {info['imagens']} imagens + {info['ineditas']} inéditas {info['instancias']}")
        console.print("[bold yellow]Lembrete:[/bold yellow] isto é só teste. A entrega precisa de imagens reais.")


@app.command()
def importar(
    origem: Path = typer.Argument(..., help="Pasta de um export YOLO (Roboflow, CVAT, LabelImg...)."),
    substituir: bool = typer.Option(False, "--substituir", help="Apaga o que já existe em dataset/bruto."),
    config: Path | None = OPCAO_CONFIG,
) -> None:
    """Copia um export no formato YOLO para dataset/bruto (juntando train/valid/test)."""
    with tratar_erros():
        cfg = carregar(config)
        info = ds.importar_pasta_yolo(cfg, origem, substituir)
        registrar(cfg, "importar", linha_de_comando())
        console.print(f"[green]{info['copiadas']} imagens importadas[/green] para {_rel(cfg, cfg.p('bruto'))}.")
        if info["rotulos_sem_imagem"]:
            console.print(f"[yellow]Aviso:[/yellow] {info['rotulos_sem_imagem']} rótulos sem imagem foram ignorados.")


@app.command("baixar-roboflow")
def baixar_roboflow(
    workspace: str = typer.Argument(..., help="Workspace no Roboflow (aparece na URL)."),
    projeto: str = typer.Argument(..., help="Nome do projeto no Roboflow."),
    versao: int = typer.Argument(..., help="Número da versão do dataset."),
    config: Path | None = OPCAO_CONFIG,
) -> None:
    """Baixa um dataset do Roboflow (formato yolov8) usando ROBOFLOW_API_KEY e importa para dataset/bruto."""
    with tratar_erros():
        cfg = carregar(config)
        info = ds.baixar_roboflow(cfg, workspace, projeto, versao)
        registrar(cfg, "baixar-roboflow", linha_de_comando())
        console.print(f"[green]{info['copiadas']} imagens importadas.[/green] Classes do export: {info['classes_do_export']}")


@app.command()
def preparar(
    forcar: bool = typer.Option(False, "--forcar", help="Segue mesmo com menos imagens que o mínimo."),
    config: Path | None = OPCAO_CONFIG,
) -> None:
    """Valida rótulos, divide 80/20 (estratificado), gera data.yaml e checa as inéditas."""
    with tratar_erros():
        from .deteccao import exemplos_anotados

        cfg = carregar(config)
        resumo = ds.preparar(cfg, forcar)
        exemplos_anotados(cfg)
        registrar(cfg, "preparar", linha_de_comando())

        tabela = Table(title="Dataset preparado")
        for coluna in ("classe", "caixas treino", "caixas val", "imagens c/ classe"):
            tabela.add_column(coluna, justify="right" if coluna != "classe" else "left")
        for i, nome in enumerate(cfg.classes):
            tabela.add_row(
                f"{i}: {nome}",
                str(resumo["treino"]["instancias"][nome]),
                str(resumo["validacao"]["instancias"][nome]),
                str(resumo["total"]["imagens_por_classe"][nome]),
            )
        console.print(tabela)
        console.print(
            f"Imagens: {resumo['treino']['imagens']} treino / {resumo['validacao']['imagens']} val "
            f"({resumo['proporcao_validacao'] * 100:.0f}% val) | inéditas: {resumo['ineditas']['quantidade']}"
        )
        _mostrar_avisos(resumo["avisos"])
        for item, ok in resumo["checklist"].items():
            console.print(f"  {'[green]OK[/green]' if ok else '[red]FALTA[/red]'}  {item.replace('_', ' ')}")
        console.print(f"data.yaml: {resumo['data_yaml']}")


# --------------------------------------------------------------------------- modelo


@app.command()
def treinar(
    epocas: int | None = typer.Option(None, "--epocas", "-e", min=1, help="Sobrescreve treino.epocas."),
    justificativa: str | None = typer.Option(None, help="Obrigatória se usar menos de 20 épocas."),
    dispositivo: str | None = typer.Option(None, help="'cpu' ou '0' (GPU)."),
    imgsz: int | None = typer.Option(None, help="Tamanho da imagem (múltiplo de 32)."),
    batch: int | None = typer.Option(None),
    config: Path | None = OPCAO_CONFIG,
) -> None:
    """Treina o YOLO a partir do modelo leve pré-treinado e guarda o best.pt em modelo/."""
    with tratar_erros():
        from .treino import treinar as executar

        cfg = carregar(config)
        ajustar_treino(cfg, epocas, justificativa, dispositivo, imgsz, batch)
        t = cfg.treino
        console.print(
            f"Treinando {t.modelo_base} por {t.epocas} épocas (imgsz={t.imgsz}, batch={t.batch}, "
            f"device={t.dispositivo}). Tire um print da tela durante o treino para o relatório!"
        )
        info = executar(cfg)
        registrar(
            cfg,
            "treinar",
            linha_de_comando(),
            f'yolo detect train model={t.modelo_base} data="{_rel(cfg, cfg.data_yaml)}" epochs={t.epocas} '
            f"imgsz={t.imgsz} batch={t.batch} device={t.dispositivo} project=\"{cfg.p('runs').as_posix()}\" name={t.nome_execucao}",
        )
        console.print(f"[green]Treino concluído[/green] em {info['duracao_s'] / 60:.1f} min ({info['epocas_executadas']} épocas).")
        console.print(f"best.pt: {info['best_pt_original']} → copiado para {info['best_pt']} ({info['best_pt_mb']} MB)")


@app.command()
def validar(
    pesos: Path | None = typer.Option(None, help="Padrão: modelo/best.pt"),
    dispositivo: str | None = typer.Option(None),
    imgsz: int | None = typer.Option(None),
    config: Path | None = OPCAO_CONFIG,
) -> None:
    """Roda a validação e salva Precision, Recall, mAP50 e mAP50-95 (geral e por classe)."""
    with tratar_erros():
        from .treino import EXPLICACAO_CURTA
        from .treino import validar as executar

        cfg = carregar(config)
        ajustar_treino(cfg, dispositivo=dispositivo)
        usar_imgsz_do_treino(cfg, imgsz)
        m = executar(cfg, pesos)
        registrar(
            cfg,
            "validar",
            linha_de_comando(),
            f'yolo detect val model="{m["pesos"]}" data="{_rel(cfg, cfg.data_yaml)}" imgsz={cfg.treino.imgsz} '
            f"project=\"{cfg.p('runs').as_posix()}\" name=val",
        )
        tabela = Table(title="Validação")
        for coluna in ("conjunto", "Precision", "Recall", "mAP50", "mAP50-95"):
            tabela.add_column(coluna, justify="left" if coluna == "conjunto" else "right")
        linhas = [("todas", m["geral"]), *m["por_classe"].items()]
        for nome, v in linhas:
            tabela.add_row(nome, *(f"{v[k]:.3f}" for k in ("precision", "recall", "mAP50", "mAP50_95")))
        console.print(tabela)
        for nome, texto in EXPLICACAO_CURTA.items():
            console.print(f"  [bold]{nome}[/bold]: {texto}")


@app.command()
def detectar(
    pesos: Path | None = typer.Option(None, help="Padrão: modelo/best.pt"),
    fonte: Path | None = typer.Option(None, help="Pasta de imagens (padrão: dataset/ineditas)."),
    conf: float | None = typer.Option(None, min=0.0, max=1.0, help="Confiança mínima (padrão 0.25)."),
    dispositivo: str | None = typer.Option(None),
    imgsz: int | None = typer.Option(None),
    config: Path | None = OPCAO_CONFIG,
) -> None:
    """Detecta laranjas nas imagens inéditas e gera painéis de inspeção."""
    with tratar_erros():
        from .deteccao import detectar as executar

        cfg = carregar(config)
        ajustar_treino(cfg, dispositivo=dispositivo)
        usar_imgsz_do_treino(cfg, imgsz)
        r = executar(cfg, pesos, fonte, conf)
        registrar(
            cfg,
            "detectar",
            linha_de_comando(),
            f'yolo detect predict model="{r["pesos"]}" source="{r["fonte"]}" conf={r["confianca_minima"]} '
            f"imgsz={cfg.treino.imgsz} save=True project=\"{cfg.p('runs').as_posix()}\" name=predict",
        )
        for img in r["imagens"]:
            dets = ", ".join(f"{d['classe']} {d['confianca']:.2f}" for d in img["deteccoes"]) or "[red]nenhuma[/red]"
            extra = ""
            if img["avaliacao"]:
                a = img["avaliacao"]
                cor = "green" if a["situacao"] == "acerto" else "red"
                extra = f"  [{cor}]{a['situacao']}[/{cor}]"
            console.print(f"  {img['arquivo']}: {dets}{extra}")
        console.print(f"[green]Painéis e mosaico em[/green] {_rel(cfg, cfg.p('resultados') / 'deteccoes')}")
        if r["total_imagens"] < cfg.divisao.minimo_ineditas:
            console.print(f"[yellow]Aviso:[/yellow] a atividade pede pelo menos {cfg.divisao.minimo_ineditas} imagens inéditas.")


# --------------------------------------------------------------------------- entrega


@app.command()
def relatorio(config: Path | None = OPCAO_CONFIG) -> None:
    """Gera relatorio.md e relatorio.pdf com os números reais dos resultados."""
    with tratar_erros():
        from .relatorio import gerar

        cfg = carregar(config)
        info = gerar(cfg)
        registrar(cfg, "relatorio", linha_de_comando())
        console.print(f"[green]Relatório:[/green] {info['pdf']} (e {info['markdown']})")
        if info["sintetico"]:
            console.print("[bold red]O relatório foi gerado com dataset SINTÉTICO. Não entregue esse PDF.[/bold red]")
        if info["etapas_faltando"]:
            console.print(f"[yellow]Seções pendentes:[/yellow] {', '.join(info['etapas_faltando'])}")
        if info["textos_para_editar"]:
            console.print(f"[yellow]Edite em configs/relatorio.yaml:[/yellow] {', '.join(info['textos_para_editar'])}")


@app.command()
def entregar(config: Path | None = OPCAO_CONFIG) -> None:
    """Monta o Projeto_Final_YOLO_<Nome>.zip e lista o que ainda falta."""
    with tratar_erros():
        from .entrega import entregar as executar

        cfg = carregar(config)
        registrar(cfg, "entregar", linha_de_comando())
        info = executar(cfg)
        console.print(f"[green]ZIP gerado:[/green] {info['zip']} ({info['tamanho_mb']} MB)")
        if info["pendencias"]:
            console.print("[bold yellow]Antes de enviar, resolva:[/bold yellow]")
            for item in info["pendencias"]:
                console.print(f"  - {item}")
        else:
            console.print("[bold green]Checklist completo. Pode enviar.[/bold green]")


@app.command()
def tudo(
    epocas: int | None = typer.Option(None, "--epocas", "-e", min=1),
    justificativa: str | None = typer.Option(None),
    dispositivo: str | None = typer.Option(None),
    imgsz: int | None = typer.Option(None),
    batch: int | None = typer.Option(None),
    forcar: bool = typer.Option(False, "--forcar"),
    config: Path | None = OPCAO_CONFIG,
) -> None:
    """Roda preparar → treinar → validar → detectar → relatorio → entregar em sequência."""
    preparar(forcar=forcar, config=config)
    treinar(epocas=epocas, justificativa=justificativa, dispositivo=dispositivo, imgsz=imgsz, batch=batch, config=config)
    validar(pesos=None, dispositivo=dispositivo, imgsz=None, config=config)
    detectar(pesos=None, fonte=None, conf=None, dispositivo=dispositivo, imgsz=None, config=config)
    relatorio(config=config)
    entregar(config=config)


if __name__ == "__main__":
    app()
