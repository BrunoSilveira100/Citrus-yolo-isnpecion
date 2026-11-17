# CLAUDE.md

Siga as regras de [AGENTS.md](AGENTS.md); elas valem para qualquer agente.

Ambiente local: Windows + PowerShell. Trabalhe de dentro de `backend/` com o Python da `.venv`
(`backend\.venv\Scripts\python.exe`). No Python global desta máquina o torch é bloqueado pelo
Controle de Aplicativo do Windows; use sempre a `.venv` ou o notebook do Colab (`frontend/notebooks/`).

Comandos úteis (de dentro de `backend/`):
- `.venv\Scripts\python.exe -m ruff check .` e `.venv\Scripts\python.exe -m pytest`
- teste ponta a ponta: veja a seção 5 de `comandos.txt` (sempre termine com `citrus-yolo sintetico --limpar`)
- treino longo: rode em segundo plano no PowerShell; não use monitor bash, porque ele manda Ctrl+C ao console
