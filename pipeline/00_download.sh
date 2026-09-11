#!/usr/bin/env bash
# Downloads the raw brain data (only the files we need) into pipeline/raw/
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw && cd raw
if [ ! -d Drosophila_brain_model ]; then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/philshiu/Drosophila_brain_model.git
  (cd Drosophila_brain_model && git checkout HEAD -- Connectivity_783.parquet Completeness_783.csv)
fi
if [ ! -d flywire_annotations ]; then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/flyconnectome/flywire_annotations.git
  (cd flywire_annotations && git checkout HEAD -- supplemental_files/Supplemental_file1_neuron_annotations.tsv)
fi
echo "raw data ready"
