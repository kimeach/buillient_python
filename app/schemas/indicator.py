from pydantic import BaseModel
from typing import List, Optional

class IndicatorRequest(BaseModel):
    close: List[float]

class IndicatorResponse(BaseModel):
    ma_20: List[Optional[float]]
    rsi_14: List[Optional[float]]
    bb_ma: List[Optional[float]]
    bb_upper: List[Optional[float]]
    bb_lower: List[Optional[float]]