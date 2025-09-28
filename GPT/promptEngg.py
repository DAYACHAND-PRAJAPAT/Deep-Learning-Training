from transformers import pipeline

model = pipeline("text-generation", model="gpt2")

query = input("Enter the query: ")

while query not in ["quit", "exit", 'q']:
    output = model(query, max_new_tokens=40)
    print(output[0]['generated_text'])
    query = input("Enter the query: ")
