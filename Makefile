install:
	uv sync

install-dev:
	uv sync --dev
	pre-commit install && pre-commit autoupdate
