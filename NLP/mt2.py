# English to German
from transformers import MarianMTModel, MarianTokenizer

print("==== Normal Translation ====")
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

sentences = [
    'Good Morning!',
    'How are you?',
    'Is your dog okay now?'
]

batch = tok(sentences, return_tensors='pt', padding=True)
translated = model.generate(**batch)

print("Translating English to German!!")
for s, t in zip(sentences, translated):
    print(f"{s} -> {tok.decode(t, skip_special_tokens=True)}")


# English to Hindi
print("==== Idioms Translation ====")
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')

sentences = [
    'Good Morning!',
    'How are you?',
    'Is your dog okay now?',
    'This is not your cup of tea',
    'Now ball is in your court'
]

batch = tok(sentences, return_tensors='pt', padding=True)
translated = model.generate(**batch)

print("Translating English to Hindi!!")
for s, t in zip(sentences, translated):
    print(f"{s} -> {tok.decode(t, skip_special_tokens=True)}")


# Reverse Translation
print("==== Reverse Translation ====")
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

sentences = [
    'Good Morning!',
    'How are you?',
    'Is your dog okay now?'
]

batch = tok(sentences, return_tensors='pt', padding=True)
translated = model.generate(**batch)
print("Translating English to German!!")
de  = []
for s, t in zip(sentences, translated):
    print(f"{s} -> {tok.decode(t, skip_special_tokens=True)}")
    de.append(tok.decode(t, skip_special_tokens=True))


model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')

sentences = de

batch = tok(sentences, return_tensors='pt', padding=True)
translated = model.generate(**batch)

print("Translating German to English!!")
for s, t in zip(sentences, translated):
    print(f"{s} -> {tok.decode(t, skip_special_tokens=True)}")
