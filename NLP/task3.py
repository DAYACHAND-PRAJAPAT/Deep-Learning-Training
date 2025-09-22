import torch
from transformers import MarianMTModel, MarianTokenizer

sentence = ["I", "love", "my", "dog"]
print("Input:", sentence)

input_text = " ".join(sentence)

model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

input_tokens = tokenizer(input_text, return_tensors="pt")
print("Tensor Conversion:", input_tokens["input_ids"])

translated_tokens = model.generate(**input_tokens)

decoded_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print("Decoded Translation:", decoded_text)

final_translation = decoded_text
print("Final Translation:", final_translation)