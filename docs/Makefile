# Makefile for the MNESIS documentation.
#
# The site is built into docs/_build/html (git-ignored); only the sources
# (conf.py, *.md, requirements.txt, this Makefile) are versioned. The GitHub
# Actions workflow .github/workflows/static.yml runs `make -C docs html` and
# deploys docs/_build/html to GitHub Pages on every push to main.
#
#   pip install -r docs/requirements.txt
#   make -C docs html        # symlink notebooks + build the site
#   open docs/_build/html/index.html

SPHINXBUILD ?= sphinx-build
SPHINXOPTS  ?=
BUILDDIR     = _build

NOTEBOOKS := $(notdir $(wildcard ../src/*.ipynb))

.PHONY: help notebooks html clean all

help:
	@echo "Available targets:"
	@echo "  notebooks  (re)create symlinks to ../src/*.ipynb into notebooks/"
	@echo "  html       build the HTML site in $(BUILDDIR)/html"
	@echo "  clean      remove generated artifacts (keeps conf.py, Makefile, *.md)"
	@echo "  all        notebooks + html (default)"

notebooks:
	@mkdir -p notebooks
	@rm -f notebooks/*.ipynb
	@for nb in $(NOTEBOOKS); do \
		ln -sf ../../src/$$nb notebooks/$$nb ; \
	done

html: notebooks
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -d $(BUILDDIR)/doctrees . $(BUILDDIR)/html
	@echo "Site ready in $(BUILDDIR)/html"

clean:
	rm -rf $(BUILDDIR) notebooks

all: html
