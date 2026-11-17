# Instruções para agentes de IA (Codex, Claude Code, Copilot...)

Projeto acadêmico: treinar um YOLO (Ultralytics) para detectar `laranja_boa` e `laranja_defeito`.

## Regras
- Código, comentários, mensagens e docs em **português do Brasil**, tom direto e didático.
- **Nunca invente métricas, prints ou best.pt.** Todo número do relatório sai de `resultados/*.json`.
- O dataset sintético (`citrus-yolo sintetico`) é só para testar o pipeline. Não entra na entrega.
- `ultralytics`/`torch` só podem ser importados **dentro de funções** (testes rodam sem torch).
- Erros esperados levantam `ErroProjeto` com mensagem acionável ("rode antes: citrus-yolo preparar").
- Caminhos: sempre via `cfg.p("nome")`, relativos à raiz do projeto. Imagens: `render.ler_imagem` /
  `render.salvar_imagem` (funcionam com acento no caminho no Windows).

## Antes de terminar qualquer mudança
```bash
ruff check .
pytest
```

## Estrutura do repositório
- `backend/`: pipeline Python (este arquivo fica aqui). Rode os comandos de dentro de `backend/`.
- `frontend/`: camada de apresentação (apresentação .docx, notebook do Colab, docs). Sem código Python.
- Na raiz só ficam `README.md`, `.github/` (CI) e `.vscode/`.

## Mapa do backend
- `src/citrus_yolo/config.py`: dataclasses + validação do `configs/projeto.yaml`
- `dataset.py`: rótulos, split estratificado, data.yaml, hashes · `sintetico.py`: dados de teste
- `treino.py`: treinar/validar · `deteccao.py`: inéditas + painel · `relatorio.py`: MD/PDF
- `entrega.py`: ZIP final · `registro.py`: log de comandos · `cli.py`: Typer
