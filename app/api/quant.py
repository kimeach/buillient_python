from fastapi import APIRouter
from app.schemas.indicator import IndicatorRequest, IndicatorResponse
from app.quant_analysis import moving_average, rsi, bollinger_bands
import pandas as pd
import math

router = APIRouter(prefix="/quant")

def to_safe_list(series):
    return [
        x if isinstance(x, (int, float)) and not math.isnan(x) else None
        for x in series
    ]
@router.post("/indicators", response_model=IndicatorResponse)
def calculate_indicators(data: IndicatorRequest):
    df = pd.DataFrame({"close": data.close})
    ma = moving_average(df, "close", 20)
    rsi_val = rsi(df, "close", 14)
    bb = bollinger_bands(df, "close", 20, 2)

    return IndicatorResponse(
        ma_20=to_safe_list(ma),
        rsi_14=to_safe_list(rsi_val),
        bb_ma=to_safe_list(bb["ma"]),
        bb_upper=to_safe_list(bb["upper"]),
        bb_lower=to_safe_list(bb["lower"]),
    )




from app.schemas.score import ScoreRequest, ScoreResponse
from app.services.score_service import calculate_score
@router.post("/score", response_model=ScoreResponse)
def quant_score(data: ScoreRequest):
    return calculate_score(data.per, data.pbr, data.roe)