#!/usr/bin/env bash
torchserve --stop

torchserve \
  --start \
  --model-store model_store \
  --models emotions_classifier=emotions_classifier.mar \
  --ts-config src/config.properties
