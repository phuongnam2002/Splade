from pyvi.ViTokenizer import tokenize
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base')
model = AutoModel.from_pretrained('VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base')

text = tokenize('trường đại học bách khoa')

inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

outputs = model(**inputs)

print(outputs)

