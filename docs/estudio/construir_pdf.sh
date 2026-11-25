#!/bin/bash
# Genera docs/RZI_Documento_de_Estudio.pdf a partir de documento.md.
# Requiere: pandoc (>= 2.x), Node.js con Playwright y Chromium.
# Las figuras se regeneran con: python scripts/generar_resultados.py
set -e
cd "$(dirname "$0")"
pandoc documento.md --standalone --mathml --toc --toc-depth=2 \
       --css estilo.css --metadata pagetitle="RZI - Documento de estudio" \
       -o documento.html
NODE_PATH="${NODE_PATH:-$(npm root -g)}" node imprimir_pdf.js "$PWD/documento.html" "$PWD/../RZI_Documento_de_Estudio.pdf"
echo "PDF generado: docs/RZI_Documento_de_Estudio.pdf"
