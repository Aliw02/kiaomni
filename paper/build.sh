#!/usr/bin/env bash
# build.sh — Compile KiaOmni_Paper.md → KiaOmni_Paper.pdf via Pandoc + LaTeX.
#
# Pipeline:
#   1. Pandoc converts ../KiaOmni_Paper.md into KiaOmni_Paper.tex using
#      template.tex and --listings (so fenced code becomes lstlisting).
#   2. pdflatex runs twice (refs/TOC). If pdflatex is missing, xelatex
#      is used as fallback.
#
# KNOWN SOURCE-SIDE GOTCHA (NOT FIXED BY TEMPLATE):
#   The source markdown wraps several math formulas inside triple-backtick
#   fenced code blocks (```...```). Pandoc treats these as code listings,
#   not as math. If you want them typeset as math, convert those fences in
#   KiaOmni_Paper.md to display-math delimiters: $$ ... $$ (or \[ ... \]).
#
# Usage:  ./build.sh

set -euo pipefail

# Resolve script dir so the script works regardless of CWD
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

INPUT_MD="../KiaOmni_Paper.md"
TEX_OUT="KiaOmni_Paper.tex"
PDF_OUT="KiaOmni_Paper.pdf"
TEMPLATE="template.tex"

if [[ ! -f "$INPUT_MD" ]]; then
    echo "ERROR: source not found: $INPUT_MD" >&2
    exit 1
fi
if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: template not found: $TEMPLATE" >&2
    exit 1
fi

# ---------- Step 1: Pandoc MD → TeX ----------
if ! command -v pandoc >/dev/null 2>&1; then
    echo "ERROR: pandoc not found in PATH" >&2
    exit 1
fi

echo "[1/3] Pandoc:  $INPUT_MD -> $TEX_OUT"
pandoc "$INPUT_MD" \
    -o "$TEX_OUT" \
    --template="$TEMPLATE" \
    --standalone \
    --from=markdown+smart+pipe_tables+backtick_code_blocks+yaml_metadata_block \
    --top-level-division=section \
    --listings \
    --number-sections \
    --toc --toc-depth=2

# ---------- Step 2: Select LaTeX compiler ----------
# Preference order (per PANDOC_AUDIT.md §8): tectonic > xelatex > pdflatex.
# Tectonic is a self-contained XeTeX engine that fetches missing packages on
# demand — works on this machine where MiKTeX could not be installed.
TECTONIC="/d/MyFolder/tools/tectonic.exe"
if [[ -x "$TECTONIC" ]] || command -v tectonic >/dev/null 2>&1; then
    LATEX=tectonic
    [[ -x "$TECTONIC" ]] && LATEX_BIN="$TECTONIC" || LATEX_BIN="tectonic"
    echo "[2/3] Using tectonic ($LATEX_BIN) — single-pass, auto-fetches packages"
elif command -v xelatex >/dev/null 2>&1; then
    LATEX=xelatex; LATEX_BIN=xelatex
    echo "[2/3] Using xelatex"
elif command -v pdflatex >/dev/null 2>&1; then
    LATEX=pdflatex; LATEX_BIN=pdflatex
    echo "[2/3] Using pdflatex (Unicode may break)"
else
    echo "ERROR: no LaTeX compiler (tectonic/xelatex/pdflatex) found" >&2
    exit 1
fi

# ---------- Step 3: Compile ----------
if [[ "$LATEX" == "tectonic" ]]; then
    # Tectonic handles multi-pass internally; --keep-logs to inspect on failure.
    echo "[3/3] $LATEX_BIN (single-pass, auto-fetching packages)"
    "$LATEX_BIN" --keep-logs --keep-intermediates "$TEX_OUT"
else
    echo "[3/3] $LATEX_BIN (pass 1/2)"
    "$LATEX_BIN" -interaction=nonstopmode -halt-on-error "$TEX_OUT" >/dev/null
    echo "      $LATEX_BIN (pass 2/2)"
    "$LATEX_BIN" -interaction=nonstopmode -halt-on-error "$TEX_OUT" >/dev/null
fi

if [[ -f "$PDF_OUT" ]]; then
    echo
    echo "OK -> $SCRIPT_DIR/$PDF_OUT"
else
    echo "ERROR: $PDF_OUT was not produced — inspect KiaOmni_Paper.log" >&2
    exit 1
fi
