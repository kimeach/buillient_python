def calculate_score(per: float, pbr: float, roe: float) -> dict:
    per_score = max(0, min(100, 100 - per * 5))
    pbr_score = max(0, min(100, 100 - pbr * 10))
    roe_score = max(0, min(100, roe * 5))

    value_score = (per_score + pbr_score) / 2
    growth_score = roe_score
    total_score = round((value_score + growth_score) / 2, 2)

    return {
        "value_score": round(value_score, 2),
        "growth_score": round(growth_score, 2),
        "total_score": total_score
    }