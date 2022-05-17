main: coverage build_docs serve

build_docs: build_readme
	PYTHONPATH=src poetry run mkdocs build

serve:
	PYTHONPATH=src poetry run mkdocs serve

lint:
	poetry run pylint balancer

black:
	poetry run black .

test:
	PYTHONPATH=src poetry run pytest tests

coverage:
	PYTHONPATH=src poetry run coverage run --source=balancer -m pytest tests && poetry run coverage report -m && poetry run coverage html

build_readme: README.ipynb
	PYTHONPATH=src poetry run jupyter nbconvert --to markdown --execute README.ipynb && rm -rf docs/README_files/ && mv README_files docs/