from pydantic import BaseModel

class ScoreRequest(BaseModel):
    per: float
    pbr: float
    roe: float

class ScoreResponse(BaseModel):
    value_score: float
    growth_score: float
    total_score: float