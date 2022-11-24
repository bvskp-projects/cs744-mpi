init:
	python -m pip install -r requirements.txt

lint:
	flake8 | head

test:
	python -m unittest -f

clean:
	rm -rf build

freeze:
	python -m pip freeze > requirements.txt

loc:
	find . -name '*.py' | xargs wc -l

.PHONY: init lint test clean freeze loc
