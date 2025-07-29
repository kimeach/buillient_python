import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import re

# CUDA 디버깅 환경변수 설정
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# GPU 메모리 정리
torch.cuda.empty_cache()

# === 1. 감성 분석 모델 (monologg/kobert) ===
print("감성 분석 모델 로딩 중...")
sentiment_model_name = "monologg/kobert"
sentiment_tokenizer = AutoTokenizer.from_pretrained(
    sentiment_model_name, 
    trust_remote_code=True,
    use_fast=False  # 안정성을 위해 fast tokenizer 비활성화
)
sentiment_model = BertForSequenceClassification.from_pretrained(
    sentiment_model_name, 
    num_labels=3
)

# GPU 사용 시 모델을 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentiment_model = sentiment_model.to(device)
print(f"사용 중인 디바이스: {device}")

# 감성 예측 함수
def predict_sentiment(text):
    try:
        sentiment_model.eval()
        
        # 텍스트 전처리 - 특수문자 및 긴 텍스트 처리
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = text.strip()
        
        if not text:
            return {"예측": "중립", "신뢰도": "0.5000", "모든_확률": {"부정": "0.3333", "중립": "0.3333", "긍정": "0.3333"}}
        
        # 토큰화 시 더 안전한 설정
        inputs = sentiment_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128,  # 길이 증가
            add_special_tokens=True
        )
        
        # GPU로 입력 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            
        label_map = {0: "부정", 1: "중립", 2: "긍정"}
        return {
            "예측": label_map[pred],
            "신뢰도": f"{torch.max(probs).item():.4f}",
            "모든_확률": {label_map[i]: f"{probs[0][i].item():.4f}" for i in range(3)}
        }
    except Exception as e:
        print(f"감성 분석 오류: {e}")
        return {"예측": "중립", "신뢰도": "0.5000", "모든_확률": {"부정": "0.3333", "중립": "0.3333", "긍정": "0.3333"}}

# === 2. 키워드 추출 모델 (KeyBERT + 다른 모델 사용) ===
print("키워드 추출 모델 로딩 중...")

# 더 안정적인 한국어 모델 사용
try:
    # 첫 번째 선택: jhgan/ko-sroberta-multitask
    kw_model = KeyBERT(model=SentenceTransformer("jhgan/ko-sroberta-multitask"))
    print("키워드 추출 모델: jhgan/ko-sroberta-multitask")
except Exception as e:
    try:
        # 두 번째 선택: paraphrase-multilingual-MiniLM-L12-v2
        kw_model = KeyBERT(model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))
        print("키워드 추출 모델: paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e2:
        print(f"키워드 추출 모델 로딩 실패: {e2}")
        kw_model = None

# 키워드 추출 함수
def extract_keywords(text, top_n=5):
    if kw_model is None:
        # 모델이 없을 경우 간단한 키워드 추출
        words = text.split()
        return words[:top_n] if len(words) >= top_n else words
    
    try:
        # 텍스트 전처리
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 10:  # 너무 짧은 텍스트는 그대로 반환
            return text.split()
        
        keywords = kw_model.extract_keywords(
            text, 
            top_n=top_n, 
            stop_words=["있다", "했다", "대한", "등", "이다", "하다", "되다", "것", "수", "때", "년", "월", "일"],
            use_mmr=True,  # 다양성을 위해 MMR 사용
            diversity=0.5
        )
        return [kw for kw, score in keywords]
    except Exception as e:
        print(f"키워드 추출 오류: {e}")
        # 오류 시 단순 분할
        words = text.split()
        return words[:top_n] if len(words) >= top_n else words

# === 3. 전체 파이프라인 ===
def analyze_news(news_text):
    print(f"📰 입력 뉴스: {news_text}\n")
    
    try:
        # 1. 전체 뉴스 감성 분석
        print("📊 전체 뉴스 감성 분석:")
        overall_sentiment = predict_sentiment(news_text)
        print(f"전체 감성: {overall_sentiment['예측']} (신뢰도: {overall_sentiment['신뢰도']})")
        print()
        
        # 2. 키워드 추출
        print("🔑 키워드 추출 중...")
        keywords = extract_keywords(news_text)
        print("추출된 키워드:", keywords)
        print()
        
        # 3. 각 키워드 감성 분석
        if keywords:
            print("📊 키워드별 감성 분석:")
            for kw in keywords:
                if kw.strip():  # 빈 키워드 제외
                    result = predict_sentiment(kw)
                    print(f" • {kw}: {result['예측']} ({result['신뢰도']})")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        print("-" * 50)

# === 4. 안전한 테스트 ===
def safe_test():
    sample_news = [
        "정부의 부동산 정책이 시장에 긍정적인 영향을 주고 있다.",
        "물가 상승과 금리 인상이 경제에 부담을 주고 있다.",
        "새로운 기술 발전으로 미래가 밝아 보인다."
    ]
    
    for i, news in enumerate(sample_news, 1):
        print(f"=== 테스트 {i} ===")
        try:
            analyze_news(news)
        except Exception as e:
            print(f"테스트 {i} 실패: {e}")
            print("-" * 50)
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# === 5. 메인 실행 ===
if __name__ == "__main__":
    print("한국어 뉴스 감성분석 시스템 시작")
    print("=" * 50)
    safe_test()