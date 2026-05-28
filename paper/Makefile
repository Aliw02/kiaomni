# Makefile — KiaOmni paper build
# Targets:
#   make pdf     -> run build.sh (Pandoc + LaTeX → KiaOmni_Paper.pdf)
#   make clean   -> remove LaTeX aux files
#   make distclean -> clean + remove generated .tex and .pdf
#   make watch   -> rebuild whenever KiaOmni_Paper.md or template.tex changes
#                   (requires `entr`; falls back to a polling loop)

SHELL := /bin/bash

SRC      := ../KiaOmni_Paper.md
TEMPLATE := template.tex
TEX      := KiaOmni_Paper.tex
PDF      := KiaOmni_Paper.pdf

.PHONY: pdf clean distclean watch

pdf: $(PDF)

$(PDF): $(SRC) $(TEMPLATE) build.sh
	@bash ./build.sh

clean:
	@rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb
	@echo "Cleaned aux files."

distclean: clean
	@rm -f $(TEX) $(PDF)
	@echo "Removed generated .tex and .pdf."

watch:
	@if command -v entr >/dev/null 2>&1; then \
	    echo "Watching with entr — Ctrl-C to stop."; \
	    echo -e "$(SRC)\n$(TEMPLATE)" | entr -c bash ./build.sh; \
	else \
	    echo "entr not found — using polling loop (2s). Ctrl-C to stop."; \
	    last=""; \
	    while true; do \
	        cur=$$(stat -c %Y $(SRC) $(TEMPLATE) 2>/dev/null | md5sum | awk '{print $$1}'); \
	        if [[ "$$cur" != "$$last" ]]; then \
	            bash ./build.sh || true; \
	            last="$$cur"; \
	        fi; \
	        sleep 2; \
	    done; \
	fi
