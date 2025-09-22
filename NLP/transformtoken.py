from transformers import AutoTokenizer

token  = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Apple's runner was running quickly towards the finish line, but he wasn't sure why."

print(token.tokenize(text))
