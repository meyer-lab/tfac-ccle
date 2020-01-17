SHELL := /bin/bash

.PHONY: clean test

flist = 1 2 3 4 5 6

all: pylint.log $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Ur requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py tfac/figures/figure%.py
	mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	mkdir -p ./output/%
	. venv/bin/activate && manubot process --content-directory=manuscripts/$*/ --output-directory=output/$*/ --log-level=WARNING

output/manuscript.html: venv output/manuscript.md style.csl
	. venv/bin/activate && pandoc \
		--from=markdown --to=html5 --filter=pandoc-fignos --filter=pandoc-eqnos --filter=pandoc-tablenos \
		--bibliography=output/$*/references.json \
		--csl=style.csl \
		--metadata link-citations=true \
		--include-after-body=common/templates/manubot/default.html \
		--include-after-body=common/templates/manubot/plugins/table-scroll.html \
		--include-after-body=common/templates/manubot/plugins/anchors.html \
		--include-after-body=common/templates/manubot/plugins/accordion.html \
		--include-after-body=common/templates/manubot/plugins/tooltips.html \
		--include-after-body=common/templates/manubot/plugins/jump-to-first.html \
		--include-after-body=common/templates/manubot/plugins/link-highlight.html \
		--include-after-body=common/templates/manubot/plugins/table-of-contents.html \
		--include-after-body=common/templates/manubot/plugins/lightbox.html \
		--mathjax \
		--variable math="" \
		--include-after-body=common/templates/manubot/plugins/math.html \
		--include-after-body=common/templates/manubot/plugins/hypothesis.html \
		--output=output/$*/manuscript.html output/$*/manuscript.md

test: venv
	. venv/bin/activate; pytest -s

coverage.xml: venv
	. venv/bin/activate; pytest --junitxml=junit.xml --cov=tfac --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc tfac > pylint.log || echo "pylint exited with $?")

clean:
	rm -rf coverage.xml junit.xml output venv
