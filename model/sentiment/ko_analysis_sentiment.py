import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime

# CUDA 디버깅 환경변수 설정
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# GPU 메모리 정리
torch.cuda.empty_cache()

@dataclass
class StockRecommendation:
    """주식 추천 결과 데이터 클래스"""
    symbol: str
    name: str
    sector: str
    sentiment_score: float
    confidence: float
    reasoning: str
    risk_level: str

# === 주식 종목 데이터베이스 ===
STOCK_DATABASE = {
    # 기술/IT 섹터
    "tech": {
        "삼성전자": {"symbol": "005930", "keywords": ["반도체", "스마트폰", "전자", "메모리", "디스플레이", "AI", "5G"]},
        "SK하이닉스": {"symbol": "000660", "keywords": ["반도체", "메모리", "DRAM", "낸드플래시", "AI", "데이터센터"]},
        "네이버": {"symbol": "035420", "keywords": ["인터넷", "검색", "클라우드", "AI", "핀테크", "커머스"]},
        "카카오": {"symbol": "035720", "keywords": ["메신저", "핀테크", "게임", "콘텐츠", "모빌리티"]},
        "LG전자": {"symbol": "066570", "keywords": ["가전", "스마트홈", "전기차", "배터리", "디스플레이"]},
    },
    # 금융 섹터
    "finance": {
        "KB금융": {"symbol": "105560", "keywords": ["은행", "금융", "대출", "금리", "핀테크", "디지털뱅킹"]},
        "신한지주": {"symbol": "055550", "keywords": ["은행", "금융", "대출", "금리", "보험"]},
        "하나금융지주": {"symbol": "086790", "keywords": ["은행", "금융", "대출", "카드", "보험"]},
    },
    # 바이오/제약 섹터
    "bio": {
        "삼성바이오로직스": {"symbol": "207940", "keywords": ["바이오", "의약품", "백신", "치료제", "제약"]},
        "셀트리온": {"symbol": "068270", "keywords": ["바이오", "항체", "치료제", "의약품", "바이오시밀러"]},
        "유한양행": {"symbol": "000100", "keywords": ["제약", "의약품", "치료제", "건강", "신약"]},
    },
    # 에너지/화학 섹터
    "energy": {
        "LG화학": {"symbol": "051910", "keywords": ["배터리", "전기차", "화학", "소재", "리튬", "ESG"]},
        "POSCO홀딩스": {"symbol": "005490", "keywords": ["철강", "리튬", "배터리소재", "친환경", "수소"]},
        "SK이노베이션": {"symbol": "096770", "keywords": ["석유화학", "배터리", "전기차", "친환경", "소재"]},
    },
    # 자동차 섹터
    "auto": {
        "현대차": {"symbol": "005380", "keywords": ["자동차", "전기차", "수소차", "모빌리티", "자율주행"]},
        "기아": {"symbol": "000270", "keywords": ["자동차", "전기차", "친환경", "모빌리티"]},
    },
    # 건설/부동산 섹터
    "construction": {
        "삼성물산": {"symbol": "028260", "keywords": ["건설", "부동산", "인프라", "플랜트", "개발"]},
        "현대건설": {"symbol": "000720", "keywords": ["건설", "부동산", "인프라", "아파트", "개발"]},
    },
    # 유통/소비재 섹터
    "retail": {
        "롯데쇼핑": {"symbol": "023530", "keywords": ["유통", "백화점", "마트", "소비", "리테일"]},
        "신세계": {"symbol": "004170", "keywords": ["백화점", "유통", "이커머스", "소비", "리테일"]},
    },
    # 엔터테인먼트 섹터
    "entertainment": {
        "HYBE": {"symbol": "352820", "keywords": ["엔터테인먼트", "K-POP", "음악", "콘텐츠", "아티스트"]},
        "SM엔터테인먼트": {"symbol": "041510", "keywords": ["엔터테인먼트", "K-POP", "음악", "아티스트"]},
    }
}

# === 기존 감성 분석 코드 (간소화) ===
print("감성 분석 모델 로딩 중...")
sentiment_model_name = "monologg/kobert"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, trust_remote_code=True, use_fast=False)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_name, num_labels=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentiment_model = sentiment_model.to(device)

print("키워드 추출 모델 로딩 중...")
try:
    kw_model = KeyBERT(model=SentenceTransformer("jhgan/ko-sroberta-multitask"))
except:
    kw_model = KeyBERT(model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))

def predict_sentiment(text):
    try:
        sentiment_model.eval()
        text = re.sub(r'[^\w\s가-힣]', ' ', text).strip()
        
        if not text:
            return {"예측": "중립", "신뢰도": 0.5, "점수": 0.0}
        
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            
        label_map = {0: "부정", 1: "중립", 2: "긍정"}
        # 감성 점수 계산 (-1 ~ 1)
        sentiment_score = (probs[0][2].item() - probs[0][0].item())
        
        return {
            "예측": label_map[pred],
            "신뢰도": confidence,
            "점수": sentiment_score
        }
    except Exception as e:
        return {"예측": "중립", "신뢰도": 0.5, "점수": 0.0}

def extract_keywords(text, top_n=10):
    try:
        text = re.sub(r'[^\w\s가-힣]', ' ', text).strip()
        if len(text) < 10:
            return text.split()
        
        keywords = kw_model.extract_keywords(
            text, top_n=top_n, 
            stop_words=["있다", "했다", "대한", "등", "이다", "하다", "되다", "것", "수", "때", "년", "월", "일"],
            use_mmr=True, diversity=0.5
        )
        return [kw for kw, score in keywords]
    except:
        return text.split()[:top_n]

# === 주식 추천 시스템 ===
class StockRecommendationEngine:
    def __init__(self):
        self.stock_db = STOCK_DATABASE
        
    def calculate_keyword_match_score(self, news_keywords: List[str], stock_keywords: List[str]) -> float:
        """뉴스 키워드와 주식 키워드 매칭 점수 계산"""
        if not news_keywords or not stock_keywords:
            return 0.0
            
        matches = 0
        total_checks = 0
        
        for news_kw in news_keywords:
            for stock_kw in stock_keywords:
                total_checks += 1
                # 키워드 포함 관계 확인
                if news_kw in stock_kw or stock_kw in news_kw:
                    matches += 1
                # 부분 매칭 (2글자 이상)
                elif len(news_kw) >= 2 and len(stock_kw) >= 2:
                    if news_kw[:2] == stock_kw[:2]:
                        matches += 0.5
        
        return matches / max(total_checks, 1) if total_checks > 0 else 0.0
    
    def determine_risk_level(self, sentiment_score: float, confidence: float) -> str:
        """리스크 레벨 결정"""
        if confidence < 0.6:
            return "높음"
        elif sentiment_score > 0.3 and confidence > 0.8:
            return "낮음"
        elif sentiment_score > 0.1:
            return "중간"
        else:
            return "높음"
    
    def recommend_stocks(self, news_text: str, top_n: int = 5) -> List[StockRecommendation]:
        """뉴스 기반 주식 추천"""
        # 1. 뉴스 감성 분석
        sentiment_result = predict_sentiment(news_text)
        overall_sentiment = sentiment_result["점수"]
        overall_confidence = sentiment_result["신뢰도"]
        
        # 2. 키워드 추출
        news_keywords = extract_keywords(news_text)
        
        # 3. 각 주식에 대해 매칭 점수 계산
        recommendations = []
        
        for sector, stocks in self.stock_db.items():
            for stock_name, stock_info in stocks.items():
                # 키워드 매칭 점수
                keyword_match_score = self.calculate_keyword_match_score(
                    news_keywords, stock_info["keywords"]
                )
                
                # 키워드가 매칭되지 않으면 스킵
                if keyword_match_score < 0.1:
                    continue
                
                # 각 키워드별 감성 분석
                keyword_sentiments = []
                for keyword in news_keywords:
                    if any(kw in keyword or keyword in kw for kw in stock_info["keywords"]):
                        kw_sentiment = predict_sentiment(keyword)
                        keyword_sentiments.append(kw_sentiment["점수"])
                
                # 최종 감성 점수 (전체 감성 + 키워드 감성 평균)
                if keyword_sentiments:
                    keyword_avg_sentiment = sum(keyword_sentiments) / len(keyword_sentiments)
                    final_sentiment_score = (overall_sentiment * 0.6 + keyword_avg_sentiment * 0.4)
                else:
                    final_sentiment_score = overall_sentiment
                
                # 최종 추천 점수 (감성 점수 * 키워드 매칭 점수 * 신뢰도)
                final_score = abs(final_sentiment_score) * keyword_match_score * overall_confidence
                
                # 추천 사유 생성
                matched_keywords = [kw for kw in news_keywords 
                                  if any(stock_kw in kw or kw in stock_kw for stock_kw in stock_info["keywords"])]
                
                reasoning = f"매칭 키워드: {', '.join(matched_keywords[:3])} | "
                reasoning += f"감성: {sentiment_result['예측']} | "
                reasoning += f"섹터: {sector}"
                
                # 리스크 레벨 결정
                risk_level = self.determine_risk_level(final_sentiment_score, overall_confidence)
                
                recommendation = StockRecommendation(
                    symbol=stock_info["symbol"],
                    name=stock_name,
                    sector=sector,
                    sentiment_score=final_sentiment_score,
                    confidence=final_score,
                    reasoning=reasoning,
                    risk_level=risk_level
                )
                
                recommendations.append(recommendation)
        
        # 신뢰도 기준으로 정렬하여 상위 N개 반환
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:top_n]

# === 분석 및 추천 통합 시스템 ===
def analyze_news_and_recommend_stocks(news_text: str, top_n: int = 5):
    """뉴스 분석 + 주식 추천 통합 함수"""
    print("=" * 60)
    print("📰 뉴스 감성 분석 및 주식 추천 시스템")
    print("=" * 60)
    print(f"📰 입력 뉴스: {news_text}\n")
    
    # 1. 기본 감성 분석
    print("📊 전체 뉴스 감성 분석:")
    overall_sentiment = predict_sentiment(news_text)
    print(f"전체 감성: {overall_sentiment['예측']} (신뢰도: {overall_sentiment['신뢰도']:.4f})")
    print(f"감성 점수: {overall_sentiment['점수']:.4f} (-1: 매우부정, 0: 중립, +1: 매우긍정)")
    print()
    
    # 2. 키워드 추출
    print("🔑 추출된 키워드:")
    keywords = extract_keywords(news_text)
    print(", ".join(keywords))
    print()
    
    # 3. 주식 추천
    print("📈 주식 종목 추천:")
    recommender = StockRecommendationEngine()
    recommendations = recommender.recommend_stocks(news_text, top_n)
    
    if not recommendations:
        print("❌ 관련 주식 종목을 찾을 수 없습니다.")
        return
    
    print(f"상위 {len(recommendations)}개 추천 종목:")
    print("-" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.name} ({rec.symbol})")
        print(f"   📍 섹터: {rec.sector}")
        print(f"   📊 감성점수: {rec.sentiment_score:.4f}")
        print(f"   🎯 신뢰도: {rec.confidence:.4f}")
        print(f"   ⚠️  리스크: {rec.risk_level}")
        print(f"   💭 추천사유: {rec.reasoning}")
        print("-" * 80)

# === 테스트 및 실행 ===
def test_stock_recommendation():
    """다양한 뉴스 시나리오 테스트"""
    test_cases = [
        {
            "title": "긍정적 기술 뉴스",
            "news": "삼성전자가 새로운 AI 반도체 기술을 개발하여 메모리 성능이 크게 향상되었다. 5G와 데이터센터 시장에서의 경쟁력이 강화될 것으로 예상된다."
        },
        {
            "title": "부정적 금융 뉴스", 
            "news": "금리 인상과 대출 규제 강화로 은행들의 수익성이 악화되고 있다. KB금융과 신한지주 등 주요 금융지주의 실적 부진이 우려된다."
        },
        {
            "title": "긍정적 바이오 뉴스",
            "news": "셀트리온의 새로운 항체 치료제가 임상 3상에서 뛰어난 효과를 보였다. 바이오시밀러 시장에서의 경쟁력이 더욱 강화될 전망이다."
        },
        {
            "title": "전기차 관련 뉴스",
            "news": "현대차그룹이 전기차 생산을 확대하고 배터리 기술 개발에 대규모 투자를 발표했다. LG화학과의 협력도 강화될 예정이다."
        },
        {
            "title": "부동산 정책 뉴스",
            "news": "정부의 부동산 규제 완화 정책으로 건설업계에 호재가 예상된다. 삼성물산과 현대건설 등 대형 건설사들의 수주 증가가 기대된다."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 테스트 케이스 {i}: {test_case['title']}")
        try:
            analyze_news_and_recommend_stocks(test_case['news'], top_n=3)
        except Exception as e:
            print(f"❌ 테스트 {i} 실패: {e}")
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n" + "="*100)

if __name__ == "__main__":
    print("🚀 감성 분석 기반 주식 추천 시스템 시작")
    test_stock_recommendation()