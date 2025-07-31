import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np

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

# === KR-FinBert-SC 감성 분석 모델 로딩 ===
print("KR-FinBert-SC 감성 분석 모델 로딩 중...")
sentiment_model_name = "snunlp/KR-FinBert-SC"
try:
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_name)
    print("✅ KR-FinBert-SC 모델 로딩 완료")
except Exception as e:
    print(f"❌ KR-FinBert-SC 로딩 실패: {e}")
    print("🔄 기본 KoBert 모델로 대체...")
    sentiment_model_name = "monologg/kobert"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, trust_remote_code=True, use_fast=False)
    sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_name, num_labels=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentiment_model = sentiment_model.to(device)

print("키워드 추출 모델 로딩 중...")
try:
    kw_model = KeyBERT(model=SentenceTransformer("jhgan/ko-sroberta-multitask"))
    print("✅ 한국어 키워드 추출 모델 로딩 완료")
except:
    print("🔄 다국어 키워드 추출 모델로 대체...")
    kw_model = KeyBERT(model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))

def predict_sentiment(text):
    """KR-FinBert-SC를 사용한 금융 도메인 감성 분석"""
    try:
        sentiment_model.eval()
        # 텍스트 전처리 (금융 도메인에 맞게 조정)
        text = re.sub(r'[^\w\s가-힣%]', ' ', text).strip()
        
        if not text:
            return {"예측": "중립", "신뢰도": 0.5, "점수": 0.0}
        
        # KR-FinBert-SC는 일반적으로 더 긴 텍스트를 처리할 수 있음
        max_length = 256 if "KR-FinBert" in sentiment_model_name else 128
        
        inputs = sentiment_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            
        # KR-FinBert-SC의 라벨 매핑 (일반적으로 0: 부정, 1: 중립, 2: 긍정)
        if "KR-FinBert" in sentiment_model_name:
            # 금융 도메인 특화 라벨 매핑
            label_map = {0: "부정", 1: "중립", 2: "긍정"}
            # 금융 도메인에서는 더 섬세한 감성 점수 계산
            sentiment_score = (probs[0][2].item() - probs[0][0].item()) * 1.2  # 금융 도메인 가중치
            # 범위를 -1 ~ 1로 제한
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        else:
            # 기본 KoBert 라벨 매핑
            label_map = {0: "부정", 1: "중립", 2: "긍정"}
            sentiment_score = (probs[0][2].item() - probs[0][0].item())
        
        return {
            "예측": label_map[pred],
            "신뢰도": confidence,
            "점수": sentiment_score,
            "모델": sentiment_model_name.split("/")[-1],  # 사용된 모델 정보
            "상세확률": {
                "부정": float(probs[0][0]),
                "중립": float(probs[0][1]), 
                "긍정": float(probs[0][2])
            }
        }
    except Exception as e:
        print(f"감성 분석 오류: {e}")
        return {"예측": "중립", "신뢰도": 0.5, "점수": 0.0, "모델": "error"}

def extract_keywords(text, top_n=10):
    """금융 도메인에 특화된 키워드 추출"""
    try:
        # 금융 도메인 특화 전처리
        text = re.sub(r'[^\w\s가-힣%]', ' ', text).strip()
        if len(text) < 10:
            return text.split()
        
        # 금융 도메인 불용어 확장
        financial_stopwords = [
            "있다", "했다", "대한", "등", "이다", "하다", "되다", "것", "수", "때", 
            "년", "월", "일", "원", "달러", "억", "조", "만", "개", "명", "통해",
            "관련", "경우", "때문", "위해", "따라", "위한", "이번", "지난", "올해",
            "내년", "최근", "현재", "오늘", "어제", "내일"
        ]
        
        keywords = kw_model.extract_keywords(
            text, 
            top_n=top_n, 
            stop_words=financial_stopwords,
            use_mmr=True, 
            diversity=0.7  # 금융 도메인에서는 다양성을 높여 더 포괄적인 키워드 추출
        )
        return [kw for kw, score in keywords]
    except Exception as e:
        print(f"키워드 추출 오류: {e}")
        return text.split()[:top_n]

# === 주식 추천 시스템 (향상된 버전) ===
class StockRecommendationEngine:
    def __init__(self):
        self.stock_db = STOCK_DATABASE
        
    def calculate_keyword_match_score(self, news_keywords: List[str], stock_keywords: List[str]) -> float:
        """향상된 키워드 매칭 점수 계산"""
        if not news_keywords or not stock_keywords:
            return 0.0
            
        matches = 0
        total_weight = 0
        
        for news_kw in news_keywords:
            for stock_kw in stock_keywords:
                # 가중치 계산 (키워드 길이 기반)
                weight = min(len(news_kw), len(stock_kw)) / 10
                total_weight += weight
                
                # 완전 일치
                if news_kw == stock_kw:
                    matches += weight * 1.0
                # 포함 관계
                elif news_kw in stock_kw or stock_kw in news_kw:
                    matches += weight * 0.8
                # 부분 매칭 (2글자 이상)
                elif len(news_kw) >= 2 and len(stock_kw) >= 2:
                    if news_kw[:2] == stock_kw[:2]:
                        matches += weight * 0.5
        
        return matches / max(total_weight, 1) if total_weight > 0 else 0.0
    
    def determine_risk_level(self, sentiment_score: float, confidence: float, keyword_match: float) -> str:
        """개선된 리스크 레벨 결정"""
        # 여러 요소를 종합한 리스크 평가
        risk_score = 0
        
        # 신뢰도 기반 리스크
        if confidence < 0.6:
            risk_score += 2
        elif confidence < 0.8:
            risk_score += 1
        
        # 감성 점수 기반 리스크
        if abs(sentiment_score) < 0.1:  # 중립에 가까움
            risk_score += 1
        elif sentiment_score < -0.5:  # 매우 부정적
            risk_score += 2
        
        # 키워드 매칭 기반 리스크
        if keyword_match < 0.2:
            risk_score += 2
        elif keyword_match < 0.4:
            risk_score += 1
        
        if risk_score >= 4:
            return "높음"
        elif risk_score >= 2:
            return "중간"
        else:
            return "낮음"
    
    def recommend_stocks(self, news_text: str, top_n: int = 5) -> List[StockRecommendation]:
        """KR-FinBert-SC 기반 주식 추천"""
        # 1. 뉴스 감성 분석 (KR-FinBert-SC 사용)
        sentiment_result = predict_sentiment(news_text)
        overall_sentiment = sentiment_result["점수"]
        overall_confidence = sentiment_result["신뢰도"]
        
        print(f"📊 전체 감성 분석 결과:")
        print(f"   모델: {sentiment_result.get('모델', 'Unknown')}")
        print(f"   감성: {sentiment_result['예측']} (점수: {overall_sentiment:.4f})")
        print(f"   신뢰도: {overall_confidence:.4f}")
        
        # 2. 키워드 추출
        news_keywords = extract_keywords(news_text, top_n=15)  # 더 많은 키워드 추출
        print(f"🔑 추출된 키워드: {', '.join(news_keywords[:10])}")
        
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
                
                # 각 키워드별 감성 분석 (금융 도메인 특화)
                keyword_sentiments = []
                for keyword in news_keywords:
                    if any(kw in keyword or keyword in kw for kw in stock_info["keywords"]):
                        # 키워드 맥락을 포함한 감성 분석
                        context_text = f"{keyword} 관련 {news_text[:100]}"
                        kw_sentiment = predict_sentiment(context_text)
                        keyword_sentiments.append(kw_sentiment["점수"])
                
                # 최종 감성 점수 (전체 감성 + 키워드 감성 가중평균)
                if keyword_sentiments:
                    keyword_avg_sentiment = np.mean(keyword_sentiments)
                    # 금융 도메인에서는 키워드 감성을 더 중요하게 고려
                    final_sentiment_score = (overall_sentiment * 0.4 + keyword_avg_sentiment * 0.6)
                else:
                    final_sentiment_score = overall_sentiment
                
                # 최종 추천 점수 계산 (개선된 공식)
                base_score = abs(final_sentiment_score) * keyword_match_score * overall_confidence
                # 금융 도메인 보너스 (긍정적 감성일 때 추가 가중치)
                if final_sentiment_score > 0.2:
                    base_score *= 1.2
                
                final_score = min(base_score, 1.0)  # 1.0으로 제한
                
                # 추천 사유 생성 (더 상세하게)
                matched_keywords = [kw for kw in news_keywords 
                                  if any(stock_kw in kw or kw in stock_kw for stock_kw in stock_info["keywords"])]
                
                reasoning = f"매칭 키워드: {', '.join(matched_keywords[:3])} | "
                reasoning += f"감성: {sentiment_result['예측']}({final_sentiment_score:+.3f}) | "
                reasoning += f"매칭도: {keyword_match_score:.3f} | "
                reasoning += f"섹터: {sector}"
                
                # 리스크 레벨 결정 (향상된 방식)
                risk_level = self.determine_risk_level(final_sentiment_score, overall_confidence, keyword_match_score)
                
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
    """KR-FinBert-SC 기반 뉴스 분석 + 주식 추천 통합 함수"""
    print("=" * 80)
    print("📰 KR-FinBert-SC 기반 뉴스 감성 분석 및 주식 추천 시스템")
    print("=" * 80)
    print(f"📰 입력 뉴스: {news_text}\n")
    
    # 1. 상세 감성 분석
    print("📊 전체 뉴스 감성 분석:")
    overall_sentiment = predict_sentiment(news_text)
    print(f"사용 모델: {overall_sentiment.get('모델', 'Unknown')}")
    print(f"전체 감성: {overall_sentiment['예측']} (신뢰도: {overall_sentiment['신뢰도']:.4f})")
    print(f"감성 점수: {overall_sentiment['점수']:.4f} (-1: 매우부정, 0: 중립, +1: 매우긍정)")
    
    # 상세 확률 출력
    if '상세확률' in overall_sentiment:
        probs = overall_sentiment['상세확률']
        print(f"상세 확률: 부정({probs['부정']:.3f}) | 중립({probs['중립']:.3f}) | 긍정({probs['긍정']:.3f})")
    print()
    
    # 2. 키워드 추출
    print("🔑 추출된 키워드:")
    keywords = extract_keywords(news_text, top_n=12)
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
    print("-" * 100)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.name} ({rec.symbol})")
        print(f"   📍 섹터: {rec.sector}")
        print(f"   📊 감성점수: {rec.sentiment_score:+.4f}")
        print(f"   🎯 신뢰도: {rec.confidence:.4f}")
        print(f"   ⚠️  리스크: {rec.risk_level}")
        print(f"   💭 추천사유: {rec.reasoning}")
        print("-" * 100)
    
    return recommendations

# === 테스트 및 실행 ===
def test_stock_recommendation():
    """다양한 뉴스 시나리오 테스트 (KR-FinBert-SC 기반)"""
    test_cases = [
        {
            "title": "긍정적 기술 뉴스",
            "news": "삼성전자가 차세대 AI 반도체 기술 개발에 성공하며 글로벌 메모리 시장에서의 지배력을 더욱 강화했다. 새로운 기술로 데이터센터 수요 급증에 대응할 수 있게 되어 주가 상승이 기대된다."
        },
        {
            "title": "부정적 금융 뉴스", 
            "news": "한국은행의 기준금리 추가 인상 전망으로 금융업계 전반에 부담이 가중되고 있다. 대출 수요 감소와 충당금 증가로 KB금융, 신한지주 등 주요 은행들의 수익성 악화가 우려된다."
        },
        {
            "title": "긍정적 바이오 뉴스",
            "news": "셀트리온의 코로나19 치료제가 FDA 승인을 받으며 글로벌 시장 진출의 발판을 마련했다. 바이오시밀러 포트폴리오 확장과 함께 매출 증대가 예상되어 투자자들의 관심이 집중되고 있다."
        },
        {
            "title": "전기차 관련 뉴스",
            "news": "현대자동차그룹이 전기차 전용 플랫폼 E-GMP 기반 신차 출시를 앞두고 있으며, LG화학과의 배터리 공급 계약을 확대했다. 친환경 모빌리티 시장에서의 경쟁력 강화로 관련 주식들의 상승이 기대된다."
        },
        {
            "title": "부동산/건설 호재 뉴스",
            "news": "정부의 주택공급 확대 정책과 재건축 규제 완화로 건설업계에 새로운 기회가 열리고 있다. 삼성물산, 현대건설 등 대형 건설사들의 수주 증가와 수익성 개선이 전망된다."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 테스트 케이스 {i}: {test_case['title']}")
        try:
            recommendations = analyze_news_and_recommend_stocks(test_case['news'], top_n=3)
            
            # 추가 분석 정보
            if recommendations:
                avg_sentiment = np.mean([rec.sentiment_score for rec in recommendations])
                print(f"📈 평균 감성 점수: {avg_sentiment:+.4f}")
                high_confidence = [rec for rec in recommendations if rec.confidence > 0.5]
                print(f"🎯 높은 신뢰도 종목 수: {len(high_confidence)}")
            
        except Exception as e:
            print(f"❌ 테스트 {i} 실패: {e}")
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n" + "="*100)

if __name__ == "__main__":
    print("🚀 KR-FinBert-SC 기반 감성 분석 주식 추천 시스템 시작")
    print(f"📱 사용 장치: {device}")
    print(f"🤖 감성 분석 모델: {sentiment_model_name}")
    test_stock_recommendation()