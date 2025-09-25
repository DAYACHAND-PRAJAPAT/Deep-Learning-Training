from datasets import load_dataset

dataset = load_dataset("Abirate/english_quotes")

print(dataset['train'][0])