poetry run black . --line-length 79
poetry run isort . --profile black

poetry run python -m pytest --verbose -vv
poetry run flake8 --ignore E203,W503 .