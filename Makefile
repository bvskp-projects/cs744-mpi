init:
	python -m pip install -r requirements.txt

test:
	python -m unittest

clean:
	rm -rf build

freeze:
	python -m pip freeze > requirements.txt

.PHONY: init test clean freeze
