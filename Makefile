PYTHON ?= python3

.PHONY: all figures slides report test clean

all: test figures slides report

figures:
	$(PYTHON) code/generate_all_figures.py

slides:
	cd Seminar_Talk && latexmk -pdf main.tex

report:
	cd Seminar_Talk/Summary && latexmk -pdf Summary.tex

test:
	$(PYTHON) -m pytest tests/

clean:
	-cd Seminar_Talk && latexmk -C main.tex
	-cd Seminar_Talk/Summary && latexmk -C Summary.tex
	rm -rf code/__pycache__ tests/__pycache__ .pytest_cache
