#!/usr/bin/env bash
# convert_and_copy.sh
# -------------------
# Copie ensuite les JSON dans ../../files/ (overwrite forcé)

set -euo pipefail                            # arrêt si erreur, vars non-déclarées interdites
command cp -f generates/embedding_train.json \
             generates/embedding_vocab.json \
             generates/embedding_eval.json \
             ../../../files/
echo "✅ copie terminées."
