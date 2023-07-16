import torch
import logging
import os
import json

from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextHandler(BaseHandler):
    def __init__(self):
        super(TextHandler, self).__init__()
        self.logger = logging.getLogger('model_log')

        self._context = None
        self.manifest = None
        self.properties = None
        self.device = None

        self.model = None
        self.tokenizer = None
        self.output_mapping = None

    def initialize(self, context):
        self.properties = context.system_properties
        self.manifest = context.manifest
        self.logger.info(f'Properties: {self.properties}')
        self.logger.info(f'Manifest: {self.manifest}')

        self.load_model()
        self.load_tokenizer()
        self.load_output_mapping()

    def load_model(self):
        model_dir = self.properties.get('model_dir')
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        if os.path.isfile(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.eval()
        else:
            raise RuntimeError(f'Could not find model {model_file}')

    def load_tokenizer(self):
        model_dir = self.properties.get('model_dir')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if not self.tokenizer:
            raise RuntimeError('Could not find tokenizer tokenizer')

    def load_output_mapping(self):
        model_dir = self.properties.get('model_dir')
        mapping_file_path = os.path.join(model_dir, 'config.json')
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as file:
                self.mapping = json.load(file)['id2label']
        else:
            raise RuntimeError('Could not find output mapping file')

    def preprocess(self, data):
        self.logger.info(f'Received {len(data)} inputs: {data}')
        tokenized_data = self.tokenizer(data, padding=True, return_tensors='pt')["input_ids"]
        return tokenized_data

    def inference(self, model_input, *args, **kwargs):
        self.logger.info(f'Running inference on: {model_input}')
        outputs = self.model.forward(model_input)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = probabilities.tolist()
        self.logger.info(f'Results: {predictions}')
        return predictions

    def postprocess(self, model_output):
        predictions = model_output
        self.logger.info(f'Labels: {predictions}')
        labels = [self.map_outputs_to_labels(prediction) for prediction in predictions]
        return [labels]

    def handle(self, requests, context):
        data = requests[0].get('body')
        if data is None:
            data = requests[0].get('data')
        data = data.get('input')

        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        output_labels = self.postprocess(model_output)
        return output_labels

    def map_outputs_to_labels(self, prediction):
        return {self.mapping[str(i)]: prediction[i] for i in range(len(prediction))}
