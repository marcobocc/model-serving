#!/usr/bin/env bash

MODELS_DIR=models

MODEL_SOURCE="SamLowe/roberta-base-go_emotions"
MODEL_NAME="emotions_classifier"
MODEL_HANDLER="src/handlers/TextHandler.py"

MODEL_PATH=$MODELS_DIR/$MODEL_NAME/model
TOKENIZER_PATH=$MODELS_DIR/$MODEL_NAME/tokenizer

python setup/download.py \
  --model_source="$MODEL_SOURCE" \
  --model_name="$MODEL_NAME" \
  --models_dir="$MODELS_DIR"

torch-model-archiver \
  --model-name "$MODEL_NAME" \
  --version 1.0 \
  --model-file "$MODEL_PATH"/pytorch_model.bin \
  --handler "$MODEL_HANDLER" \
  --extra-files "$MODEL_PATH/config.json,$TOKENIZER_PATH/special_tokens_map.json,$TOKENIZER_PATH/tokenizer.json,$TOKENIZER_PATH/tokenizer_config.json,$TOKENIZER_PATH/vocab.json" \
  --export-path model_store \
  --force
