
# Desafio_R3 — FastAPI  + ML (LightGBM LTR)

## Requisitos
- Python **3.12**
- **uv** (https://docs.astral.sh/uv/)
  - Windows (PowerShell): `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - macOS: `brew install uv`
  - Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Instalación
```bash
uv sync

uv run fastapi dev app/api.py
