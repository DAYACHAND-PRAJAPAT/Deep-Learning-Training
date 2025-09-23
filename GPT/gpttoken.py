from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = '''The Wikipedia article on "Sex" explains that the term generally refers to an organism's assigned biological sex (e.g., male, female), which is distinct from "gender," a related but separate concept encompassing social roles (gender role) and personal identity (gender identity). The article also notes that "sex" can be used in other contexts, such as "human sexuality," which refers to a person's experience and expression of sexual feelings and behaviors, including their biological, psychological, physical, and emotional aspects. '''

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
print("Tokens: ",tokens)
print("IDs :", ids)
print("Decode IDs :", tokenizer.decode(ids))