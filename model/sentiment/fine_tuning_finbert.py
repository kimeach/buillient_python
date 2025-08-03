import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# 하이퍼파라미터 설정
MODEL_NAME = "snunlp/KR-FinBert-SC"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
SAVE_PATH = "./finbert_finetuned"

# 데이터 불러오기
df = pd.read_csv("train_data.csv")  # 위에서 말한 csv 파일
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 커스텀 Dataset 클래스
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

# 모델 불러오기
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Trainer 객체로 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# 모델 저장
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"📦 파인튜닝된 모델 저장 완료: {SAVE_PATH}")
