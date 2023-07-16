from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_source',
                    dest='model_source',
                    type=str,
                    help='HuggingFace model name, example: SamLowe/roberta-base-go_emotions')
parser.add_argument('--model_name',
                    dest='model_name',
                    type=str,
                    help='Name that will be given to the model')
parser.add_argument('--models_dir',
                    dest='models_dir',
                    type=str,
                    help='Directory where the model files will be downloaded')
args = parser.parse_args()

model_source = args.model_source
model_name = args.model_name
models_dir = args.models_dir

tokenizer = AutoTokenizer.from_pretrained(model_source)
model = AutoModelForSequenceClassification.from_pretrained(model_source)

tokenizer.save_pretrained(f'./{models_dir}/{model_name}/tokenizer')
model.save_pretrained(f'./{models_dir}/{model_name}/model')
