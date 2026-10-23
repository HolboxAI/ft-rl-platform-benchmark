#!/bin/bash
# Download the CholecT50 SFT checkpoint onto the instance from presigned S3 URLs.
# The URLs live in /home/ubuntu/urls.sh (scp'd separately, kept off-disk after use).
set -euo pipefail

MODEL_DIR=/home/ubuntu/model/cholec-sft-qwen3.5-4b
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# shellcheck disable=SC1091
source /home/ubuntu/urls.sh

declare -A MAP=(
  [config.json]=URL_config_json
  [model.safetensors]=URL_model_safetensors
  [model.safetensors.index.json]=URL_model_safetensors_index_json
  [chat_template.jinja]=URL_chat_template_jinja
  [generation_config.json]=URL_generation_config_json
  [preprocessor_config.json]=URL_preprocessor_config_json
  [tokenizer.json]=URL_tokenizer_json
  [tokenizer_config.json]=URL_tokenizer_config_json
  [vocab.json]=URL_vocab_json
  [merges.txt]=URL_merges_txt
  [video_preprocessor_config.json]=URL_video_preprocessor_config_json
)

for f in "${!MAP[@]}"; do
  var="${MAP[$f]}"
  url="${!var}"
  echo "downloading $f ..."
  curl -sS -f -o "$f" "$url"
done

echo "=== downloaded ==="
ls -la "$MODEL_DIR"
du -sh "$MODEL_DIR"
