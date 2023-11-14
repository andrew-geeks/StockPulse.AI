from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("KernAI/stock-news-distilbert")

tokenizer = AutoTokenizer.from_pretrained("KernAI/stock-news-distilbert")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("Tata Motors gives up early gains despite deal to pick up 27% in Freight Tiger")

print(result)