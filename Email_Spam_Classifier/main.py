import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
import string

try:
    df = pd.read_csv("spam.csv", encoding="latin-1")
except FileNotFoundError:
    print("Error: spam.csv not found. Please download the dataset and place it in the same directory.")
    exit()

df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df.drop_duplicates(inplace=True)

def preprocess_text(text):
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

x_train, x_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec, y_train)

pred = model.predict(x_test_vec)
print(f"Accuracy: {accuracy_score(y_test, pred)}")
print(f"Classification Report:\n{classification_report(y_test, pred)}")

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")