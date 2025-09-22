from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print(summarizer("The new phone release was impressive with better speed and battery life.", 
                 max_length=40, min_length=10)[0]['summary_text'])
