# analyze_sentiment.py

import json
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# 모델 로딩
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 입력 뉴스 불러오기
with open("input/news.json", "r", encoding="utf-8") as f:
    news_items = json.load(f)

results = []

for item in news_items:
    text = item["text"]
    sentiment = nlp(text)[0]
    results.append({
        "text": text,
        "label": sentiment["label"],
        "score": sentiment["score"]
    })

# 결과 저장
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ 감성 분석 완료. 결과 → output.json")
