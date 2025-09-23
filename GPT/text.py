from transformers import AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator = pipeline("text-generation", model="gpt2")

text = "You are a GPT model so try running.."

# tokens = tokenizer.tokenize(text)
# ids = tokenizer.encode(text)
# print("Tokens: ",tokens)
# print("IDs :", ids)
# print("Decode IDs :", tokenizer.decode(ids))

output = generator(text, max_length=30, num_return_sequences=1)
print("GPT: ", output)