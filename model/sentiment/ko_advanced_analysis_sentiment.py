# === 고급 기능 확장 ===
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
import yfinance as yf  # pip install yfinance
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from ko_analysis_sentiment import StockRecommendationEngine, predict_sentiment

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'

class AdvancedStockRecommendationEngine(StockRecommendationEngine):
    """고급 주식 추천 엔진"""
    
    def __init__(self):
        super().__init__()
        self.sentiment_history = []  # 감성 분석 히스토리
        
    def get_stock_price_data(self, symbol: str, period: str = "1mo") -> Dict:
        """실시간 주식 가격 데이터 조회 (야후 파이낸스)"""
        try:
            # 한국 주식은 .KS 접미사 필요
            ticker = f"{symbol}.KS"
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {"error": "데이터 없음"}
            
            current_price = hist['Close'][-1]
            prev_price = hist['Close'][0]
            change_percent = ((current_price - prev_price) / prev_price) * 100
            
            return {
                "current_price": current_price,
                "change_percent": change_percent,
                "volume": hist['Volume'][-1],
                "high_52w": hist['High'].max(),
                "low_52w": hist['Low'].min(),
                "data": hist
            }
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict:
        """기술적 지표 계산"""
        try:
            close = price_data['Close']
            
            # 이동평균
            ma5 = close.rolling(window=5).mean().iloc[-1]
            ma20 = close.rolling(window=20).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # RSI (Relative Strength Index) 계산 (과매수/과매도 지표)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # 볼린저 밴드 (Bollinger Bands)
            bb_center = close.rolling(window=20).mean().iloc[-1]
            bb_std = close.rolling(window=20).std().iloc[-1]
            bb_upper = bb_center + (bb_std * 2)
            bb_lower = bb_center - (bb_std * 2)
            
            return {
                "ma5": ma5,
                "ma20": ma20,
                "rsi": rsi,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_position": (current_price - bb_lower) / (bb_upper - bb_lower)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def sentiment_trend_analysis(self, news_list: List[str], days: int = 7) -> Dict:
        """여러 뉴스의 감성 트렌드 분석"""
        sentiment_scores = []
        timestamps = []
        
        for i, news in enumerate(news_list):
            sentiment = predict_sentiment(news)
            sentiment_scores.append(sentiment["점수"])
            timestamps.append(datetime.now() - timedelta(days=days-i))
            
            # 히스토리 저장
            self.sentiment_history.append({
                "timestamp": timestamps[-1],
                "news": news[:50] + "...",
                "sentiment_score": sentiment["점수"],
                "confidence": sentiment["신뢰도"]
            })
        
        # 트렌드 계산
        if len(sentiment_scores) >= 2:
            trend = "상승" if sentiment_scores[-1] > sentiment_scores[0] else "하락"
            trend_strength = abs(sentiment_scores[-1] - sentiment_scores[0])
        else:
            trend = "불명"
            trend_strength = 0
        
        return {
            "scores": sentiment_scores,
            "timestamps": timestamps,
            "trend": trend,
            "trend_strength": trend_strength,
            "average_sentiment": sum(sentiment_scores) / len(sentiment_scores)
        }
    
    def enhanced_recommend_stocks(self, news_text: str, include_technical: bool = True, top_n: int = 5) -> List[Dict]:
        """기술적 분석을 포함한 고급 주식 추천"""
        # 기본 추천 실행
        basic_recommendations = self.recommend_stocks(news_text, top_n * 2)  # 더 많이 가져와서 필터링
        
        enhanced_recommendations = []
        
        for rec in basic_recommendations:
            enhanced_rec = {
                "basic_info": rec,
                "price_data": None,
                "technical_indicators": None,
                "final_score": rec.confidence,
                "investment_signal": "보류"
            }
            
            if include_technical:
                # 실시간 가격 데이터 조회
                price_data = self.get_stock_price_data(rec.symbol)
                
                if "error" not in price_data:
                    enhanced_rec["price_data"] = price_data
                    
                    # 기술적 지표 계산
                    technical_indicators = self.calculate_technical_indicators(price_data["data"])
                    enhanced_rec["technical_indicators"] = technical_indicators
                    
                    # 종합 점수 계산 (감성 + 기술적)
                    if "error" not in technical_indicators:
                        technical_score = self.calculate_technical_score(technical_indicators)
                        enhanced_rec["final_score"] = (rec.confidence * 0.7 + technical_score * 0.3)
                        
                        # 투자 신호 결정
                        enhanced_rec["investment_signal"] = self.determine_investment_signal(
                            rec.sentiment_score, technical_indicators
                        )
            
            enhanced_recommendations.append(enhanced_rec)
        
        # 최종 점수 기준으로 재정렬
        enhanced_recommendations.sort(key=lambda x: x["final_score"], reverse=True)
        return enhanced_recommendations[:top_n]
    
    def calculate_technical_score(self, indicators: Dict) -> float:
        """기술적 지표 기반 점수 계산"""
        if "error" in indicators:
            return 0.5
        
        score = 0.5  # 기본 점수
        
        # RSI 점수
        rsi = indicators["rsi"]
        if 30 <= rsi <= 70:  # 정상 범위
            score += 0.2
        elif rsi < 30:  # 과매도
            score += 0.3
        elif rsi > 70:  # 과매수
            score -= 0.1
        
        # 볼린저 밴드 점수
        bb_pos = indicators["bb_position"]
        if 0.2 <= bb_pos <= 0.8:  # 정상 범위
            score += 0.2
        elif bb_pos < 0.2:  # 하단 근처 (매수 기회)
            score += 0.3
        
        # 이동평균 점수
        if indicators["ma5"] > indicators["ma20"]:  # 상승 추세
            score += 0.1
        
        return min(max(score, 0), 1)  # 0-1 범위로 제한
    
    def determine_investment_signal(self, sentiment_score: float, technical_indicators: Dict) -> str:
        """투자 신호 결정"""
        if "error" in technical_indicators:
            return "데이터부족"
        
        # 감성 점수와 기술적 지표 종합 판단
        positive_signals = 0
        negative_signals = 0
        
        # 감성 신호
        if sentiment_score > 0.2:
            positive_signals += 2
        elif sentiment_score < -0.2:
            negative_signals += 2
        
        # RSI 신호
        rsi = technical_indicators["rsi"]
        if rsi < 30:
            positive_signals += 1
        elif rsi > 70:
            negative_signals += 1
        
        # 볼린저 밴드 신호
        bb_pos = technical_indicators["bb_position"]
        if bb_pos < 0.2:
            positive_signals += 1
        elif bb_pos > 0.8:
            negative_signals += 1
        
        # 이동평균 신호
        if technical_indicators["ma5"] > technical_indicators["ma20"]:
            positive_signals += 1
        else:
            negative_signals += 1
        
        # 최종 신호 결정
        if positive_signals >= 3:
            return "매수"
        elif negative_signals >= 3:
            return "매도"
        else:
            return "보류"
    
    def generate_investment_report(self, news_text: str, top_n: int = 3) -> str:
        """투자 리포트 생성"""
        recommendations = self.enhanced_recommend_stocks(news_text, include_technical=True, top_n=top_n)
        
        report = f"""
🔍 투자 분석 리포트
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
분석 뉴스: {news_text[:100]}...

📊 전체 시장 감성 분석:
{predict_sentiment(news_text)}

💼 추천 종목 상세 분석:
"""
        
        for i, rec_data in enumerate(recommendations, 1):
            rec = rec_data["basic_info"]
            price_info = rec_data.get("price_data", {})
            tech_info = rec_data.get("technical_indicators", {})
            
            report += f"""
{'='*50}
{i}. {rec.name} ({rec.symbol})
{'='*50}
📍 기본 정보:
   - 섹터: {rec.sector}
   - 감성 점수: {rec.sentiment_score:.4f}
   - 리스크 레벨: {rec.risk_level}
   - 추천 사유: {rec.reasoning}

"""
            
            if price_info and "error" not in price_info:
                report += f"""📈 주가 정보:
   - 현재가: {price_info['current_price']:,.0f}원
   - 변동률: {price_info['change_percent']:+.2f}%
   - 거래량: {price_info['volume']:,}주
   - 52주 최고가: {price_info['high_52w']:,.0f}원
   - 52주 최저가: {price_info['low_52w']:,.0f}원

"""
            
            if tech_info and "error" not in tech_info:
                report += f"""📊 기술적 분석:
   - 5일 이평선: {tech_info['ma5']:,.0f}원
   - 20일 이평선: {tech_info['ma20']:,.0f}원
   - RSI: {tech_info['rsi']:.1f}
   - 볼린저밴드 위치: {tech_info['bb_position']:.2f} (0~1)

"""
            
            report += f"""🎯 투자 신호: {rec_data['investment_signal']}
💯 최종 점수: {rec_data['final_score']:.4f}

"""
        
        return report

# === 포트폴리오 관리 시스템 ===
class PortfolioManager:
    """포트폴리오 관리 클래스"""
    
    def __init__(self):
        self.portfolio = {}
        self.transaction_history = []
    
    def add_stock(self, symbol: str, name: str, quantity: int, purchase_price: float):
        """종목 추가"""
        if symbol in self.portfolio:
            # 기존 종목의 경우 평균 단가 계산
            existing = self.portfolio[symbol]
            total_cost = existing['quantity'] * existing['avg_price'] + quantity * purchase_price
            total_quantity = existing['quantity'] + quantity
            avg_price = total_cost / total_quantity
            
            self.portfolio[symbol].update({
                'quantity': total_quantity,
                'avg_price': avg_price
            })
        else:
            self.portfolio[symbol] = {
                'name': name,
                'quantity': quantity,
                'avg_price': purchase_price,
                'purchase_date': datetime.now()
            }
        
        # 거래 기록
        self.transaction_history.append({
            'date': datetime.now(),
            'action': '매수',
            'symbol': symbol,
            'quantity': quantity,
            'price': purchase_price
        })
    
    def remove_stock(self, symbol: str, quantity: int, sell_price: float):
        """종목 판매"""
        if symbol not in self.portfolio:
            return False
        
        if self.portfolio[symbol]['quantity'] < quantity:
            return False
        
        self.portfolio[symbol]['quantity'] -= quantity
        
        if self.portfolio[symbol]['quantity'] == 0:
            del self.portfolio[symbol]
        
        # 거래 기록
        self.transaction_history.append({
            'date': datetime.now(),
            'action': '매도',
            'symbol': symbol,
            'quantity': quantity,
            'price': sell_price
        })
        
        return True
    
    def get_portfolio_status(self) -> Dict:
        """포트폴리오 현황 조회"""
        if not self.portfolio:
            return {"message": "보유 종목이 없습니다."}
        
        total_value = 0
        total_cost = 0
        status = {}
        
        for symbol, info in self.portfolio.items():
            # 현재 가격 조회 (실제로는 API 호출)
            current_price = info['avg_price'] * 1.05  # 임시로 5% 상승 가정
            
            current_value = current_price * info['quantity']
            cost = info['avg_price'] * info['quantity']
            profit_loss = current_value - cost
            profit_loss_rate = (profit_loss / cost) * 100
            
            total_value += current_value
            total_cost += cost
            
            status[symbol] = {
                'name': info['name'],
                'quantity': info['quantity'],
                'avg_price': info['avg_price'],
                'current_price': current_price,
                'current_value': current_value,
                'profit_loss': profit_loss,
                'profit_loss_rate': profit_loss_rate
            }
        
        total_profit_loss = total_value - total_cost
        total_profit_loss_rate = (total_profit_loss / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'stocks': status,
            'summary': {
                'total_cost': total_cost,
                'total_value': total_value,
                'total_profit_loss': total_profit_loss,
                'total_profit_loss_rate': total_profit_loss_rate
            }
        }

# === 실시간 모니터링 시스템 ===
class NewsMonitoringSystem:
    """뉴스 모니터링 및 알림 시스템"""
    
    def __init__(self):
        self.watchlist = []  # 관심 종목 목록
        self.alert_thresholds = {}  # 알림 임계값
        self.recommender = AdvancedStockRecommendationEngine()
    
    def add_to_watchlist(self, symbol: str, name: str, sentiment_threshold: float = 0.3):
        """관심 종목 추가"""
        self.watchlist.append({
            'symbol': symbol,
            'name': name,
            'added_date': datetime.now()
        })
        self.alert_thresholds[symbol] = sentiment_threshold
    
    def analyze_news_batch(self, news_list: List[str]) -> Dict:
        """여러 뉴스 일괄 분석"""
        results = []
        
        for news in news_list:
            # 각 뉴스에 대해 감성 분석 및 종목 추천
            sentiment = predict_sentiment(news)
            recommendations = self.recommender.recommend_stocks(news, top_n=3)
            
            # 관심 종목 체크
            watchlist_alerts = []
            for watch_item in self.watchlist:
                for rec in recommendations:
                    if rec.symbol == watch_item['symbol']:
                        if abs(rec.sentiment_score) >= self.alert_thresholds[rec.symbol]:
                            watchlist_alerts.append({
                                'stock': rec.name,
                                'sentiment_score': rec.sentiment_score,
                                'alert_type': '긍정' if rec.sentiment_score > 0 else '부정'
                            })
            
            results.append({
                'news': news[:100] + "...",
                'timestamp': datetime.now(),
                'sentiment': sentiment,
                'recommendations': recommendations,
                'watchlist_alerts': watchlist_alerts
            })
        
        return {'analysis_results': results}

# === 웹 대시보드용 시각화 함수 ===
def create_sentiment_chart(sentiment_scores: List[float], timestamps: List[datetime]) -> str:
    """감성 점수 차트 생성 (HTML)"""
    html_chart = f"""
    <div style="width: 100%; height: 300px; border: 1px solid #ccc; padding: 20px;">
        <h3>감성 점수 추이</h3>
        <svg width="100%" height="250">
            <!-- 차트 영역 -->
    """
    
    if len(sentiment_scores) >= 2:
        width = 600
        height = 200
        
        # 점들 그리기
        for i, (score, timestamp) in enumerate(zip(sentiment_scores, timestamps)):
            x = (i / (len(sentiment_scores) - 1)) * width
            y = height - ((score + 1) / 2 * height)  # -1~1을 0~height로 변환
            
            html_chart += f'<circle cx="{x}" cy="{y}" r="4" fill="blue" />'
            
            # 선 그리기
            if i > 0:
                prev_x = ((i-1) / (len(sentiment_scores) - 1)) * width
                prev_y = height - ((sentiment_scores[i-1] + 1) / 2 * height)
                html_chart += f'<line x1="{prev_x}" y1="{prev_y}" x2="{x}" y2="{y}" stroke="blue" stroke-width="2" />'
    
    html_chart += """
        </svg>
    </div>
    """
    
    return html_chart

# === 통합 실행 함수 ===
def comprehensive_analysis_demo():
    """종합 분석 데모"""
    print("🚀 고급 주식 분석 시스템 데모")
    print("=" * 80)
    
    # 1. 고급 추천 엔진 테스트
    recommender = AdvancedStockRecommendationEngine()
    
    test_news = "삼성전자가 AI 반도체 기술의 혁신적 발전을 이루며 글로벌 시장에서의 경쟁력을 크게 강화했다. 새로운 메모리 기술로 데이터센터 시장 점유율 확대가 기대된다."
    
    print("📊 고급 분석 결과:")
    enhanced_recommendations = recommender.enhanced_recommend_stocks(test_news, top_n=3)
    
    for i, rec_data in enumerate(enhanced_recommendations, 1):
        rec = rec_data["basic_info"]
        print(f"\n{i}. {rec.name} ({rec.symbol})")
        print(f"   감성 점수: {rec.sentiment_score:.4f}")
        print(f"   최종 점수: {rec_data['final_score']:.4f}")
        print(f"   투자 신호: {rec_data['investment_signal']}")
        print(f"   리스크: {rec.risk_level}")
    
    print("\n" + "="*80)
    
    # 2. 투자 리포트 생성
    print("📋 투자 리포트:")
    report = recommender.generate_investment_report(test_news)
    print(report)
    
    # 3. 포트폴리오 관리 데모
    print("💼 포트폴리오 관리 데모:")
    portfolio = PortfolioManager()
    portfolio.add_stock("005930", "삼성전자", 10, 75000)
    portfolio.add_stock("000660", "SK하이닉스", 5, 120000)
    
    status = portfolio.get_portfolio_status()
    print("현재 포트폴리오:")
    for symbol, info in status['stocks'].items():
        print(f"- {info['name']}: {info['quantity']}주, "
              f"수익률 {info['profit_loss_rate']:+.2f}%")
    
    print(f"총 수익률: {status['summary']['total_profit_loss_rate']:+.2f}%")
    
    print("\n" + "="*80)
    
    # 4. 뉴스 모니터링 시스템 데모
    print("📰 뉴스 모니터링 시스템:")
    monitor = NewsMonitoringSystem()
    monitor.add_to_watchlist("005930", "삼성전자")
    monitor.add_to_watchlist("035420", "네이버")
    
    batch_news = [
        "코스피가 6영업일 연속 상승하며 3,254.47pt로 장을 마감했고, 이는 약 4년 만의 최고치입니다",
        "외국인과 기관이 동반 순매수하며 상승세를 견인했으며, 특히 외국인은 5,800억 원, 기관은 3,300억 원 규모 매수에 나섰습니다",
        "삼성전자가 약 2.8% 상승하며 지수 상승의 선봉 역할을 했고, 기아(4.5%), 현대차, SK하이닉스, LG에너지솔루션 등도 강세를 보였습니다",
        "반면 코스닥은 약세로 마감했으며 일부 바이오·IT 종목만 약보합세 이상 유지",
        "시장 상승세는 한미 관세 협상 타결 기대감과 반도체 및 수출기업 수주 호재, 정부의 증시 개선 정책 기대로 이어졌습니다"
    ]
    
    batch_results = monitor.analyze_news_batch(batch_news)
    print(f"분석된 뉴스 수: {len(batch_results['analysis_results'])}")
    
    for result in batch_results['analysis_results']:
        if result['watchlist_alerts']:
            print(f"⚠️ 알림: {result['watchlist_alerts'][0]['stock']} - {result['watchlist_alerts'][0]['alert_type']}")

if __name__ == "__main__":
    comprehensive_analysis_demo()