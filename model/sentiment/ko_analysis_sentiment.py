import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import json

# 1. 토크나이저 & 모델 불러오기
model_name = "monologg/kobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 2. 학습 데이터셋 준비
texts = [
    "코스피가 상승했다.",
    "경제 전망이 불투명하다.",
    "코스닥은 보합세를 보였다."
]
labels = [2, 0, 1]  # 0=부정, 1=중립, 2=긍정
dataset = Dataset.from_dict({'text': texts, 'label': labels})

def tokenize_fn(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 3. 평가 데이터셋 준비
eval_texts = [
    "경제성장률이 상승했다.",
    "물가가 폭등했다.",
    "기업 실적이 예상보다 나쁘다."
]
eval_labels = [2, 0, 0]
eval_dataset = Dataset.from_dict({'text': eval_texts, 'label': eval_labels})
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 4. 평가 지표 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {'accuracy': acc, 'f1': f1}

# 5. 학습 설정 (save_strategy를 no로 설정)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="no"  # 자동 저장 비활성화
)

# 6. Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. 학습 실행
trainer.train()

# 8. 수동 저장 (토크나이저 저장 문제 해결)
save_directory = "./best_model"
os.makedirs(save_directory, exist_ok=True)

# 모델만 저장
model.save_pretrained(save_directory)

# 토크나이저 정보를 수동으로 저장 (원본 모델명 저장)
tokenizer_info = {
    "model_name": model_name,
    "trust_remote_code": True,
    "max_length": 64
}

with open(os.path.join(save_directory, "tokenizer_info.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer_info, f, ensure_ascii=False, indent=2)

print(f"✅ 모델이 {save_directory}에 저장되었습니다.")
print("⚠️  토크나이저는 원본 모델명으로 불러와야 합니다.")

# 9. 평가 실행
eval_result = trainer.evaluate()
print("📊 평가 결과:", eval_result)

# 10. 저장된 모델 불러오기 함수
def load_saved_model(save_directory):
    """저장된 모델과 토크나이저를 불러오는 함수"""
    # 토크나이저 정보 불러오기
    tokenizer_info_path = os.path.join(save_directory, "tokenizer_info.json")
    
    if os.path.exists(tokenizer_info_path):
        with open(tokenizer_info_path, "r", encoding="utf-8") as f:
            tokenizer_info = json.load(f)
        
        # 원본 모델명으로 토크나이저 불러오기
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_info["model_name"], 
            trust_remote_code=tokenizer_info["trust_remote_code"]
        )
    else:
        # 토크나이저 정보가 없으면 원본 모델명 사용
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 학습된 모델 불러오기
    loaded_model = BertForSequenceClassification.from_pretrained(save_directory)
    
    return loaded_tokenizer, loaded_model

# 11. 모델과 토크나이저 불러오기
loaded_tokenizer, loaded_model = load_saved_model(save_directory)

# 12. 추론 함수 정의
def predict(text, model=loaded_model, tokenizer=loaded_tokenizer):
    """감정 예측 함수"""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    
    label_map = {0: "부정", 1: "중립", 2: "긍정"}
    return {
        "예측": label_map[pred],
        "신뢰도": f"{confidence:.4f}",
        "모든_확률": {label_map[i]: f"{probs[0][i].item():.4f}" for i in range(3)}
    }

# 13. 예측 테스트
test_sentences = [
    "정부의 경제정책이 긍정적으로 평가받고 있다.",
    "주식시장이 크게 하락했다.",
    "오늘 날씨가 흐리다."
]

print("\n🔍 예측 결과:")
for sentence in test_sentences:
    result = predict(sentence)
    print(f"문장: {sentence}")
    print(f"결과: {result}")
    print("-" * 50)

# 14. 모델 성능 요약
print(f"\n📈 최종 평가 성능:")
print(f"정확도: {eval_result['eval_accuracy']:.4f}")
print(f"F1 점수: {eval_result['eval_f1']:.4f}")
print(f"손실: {eval_result['eval_loss']:.4f}")