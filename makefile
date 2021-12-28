SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard tfac/figures/figure*.py)

all: $(patsubst tfac/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: tfac/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

test:
	poetry run pytest -s -x -v

coverage.xml:
	poetry run pytest --junitxml=junit.xml --cov=tfac --cov-report xml:coverage.xml

clean:
	rm -rf coverage.xml junit.xml output
