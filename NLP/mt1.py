from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

sentences = ["Good morning!", "I love programming.", "How are you today?"]

batch = tok(sentences, return_tensors="pt", padding=True)
translated = model.generate(**batch)

for s, t in zip(sentences, translated):
    print(f"{s}  -->  {tok.decode(t, skip_special_tokens=True)}")
