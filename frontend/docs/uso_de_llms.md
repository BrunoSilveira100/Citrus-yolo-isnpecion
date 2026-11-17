# Uso de LLMs no projeto

Usei assistentes de IA (LLMs) como apoio de programação, do mesmo jeito que se usa documentação ou
um colega mais experiente. O que cada um fez fica registrado aqui por transparência.

## Ferramentas
- **Claude Code (Anthropic)**, dentro do VS Code: montou a estrutura do repositório, escreveu a
  primeira versão do código e dos testes e rodou o teste de ponta a ponta com dados sintéticos.
- **Codex / outros agentes**: podem ser usados no mesmo repositório. As regras para qualquer agente
  ficam em [AGENTS.md](../../backend/AGENTS.md) e [CLAUDE.md](../../backend/CLAUDE.md).

## O que ficou com a IA
- Estrutura do projeto (backend/frontend), CLI, validações e testes.
- Gerador de laranjas sintéticas, usado só para provar que o pipeline roda.
- Busca do dataset público (Orange Fruit Diseases, Mendeley) e o script de anotação
  semi-automática (`backend/scripts/anotar_mendeley.py`), com revisão visual das caixas por amostragem.
- Execução do treino, da validação e da detecção na minha máquina.
- Rascunhos dos textos em `backend/configs/relatorio.yaml`, escritos a partir dos números reais.

## O que é responsabilidade minha
- Escolher o tema e conferir se o dataset e as anotações fazem sentido.
- Rodar o treino, tirar os prints e conferir os resultados.
- Escrever com as minhas palavras as respostas e a explicação das métricas.
- Revisar o relatório final antes de entregar.

## Regras combinadas com os agentes
1. Nenhum número é inventado: métricas, épocas e detecções saem dos arquivos de `resultados/`.
2. O dataset sintético nunca vai para a entrega (o projeto avisa em vermelho se estiver em uso).
3. Toda mudança passa por `ruff check .` e `pytest` antes de ser aceita.

## Prompt base usado
> Atue como desenvolvedor sênior em Visão Computacional (YOLO/Ultralytics). Monte um projeto completo
> para a Atividade Prática Final de IA, com organização profissional: detectar laranjas e
> classificá-las em `laranja_boa` e `laranja_defeito`, com CLI, config YAML validada, testes e
> geração automática de relatório e ZIP. Código e textos em português, sem inventar métricas.
