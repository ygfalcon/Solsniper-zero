import os
from .portfolio import calculate_order_size, Portfolio

async def evaluate(token: str, portfolio: Portfolio):
    """Return a simple buy action scaled by portfolio value and risk."""
    balance = portfolio.total_value({})
    alloc = portfolio.percent_allocated(token, {})
    rt = float(os.getenv("RISK_TOLERANCE", "0.1"))
    max_alloc = float(os.getenv("MAX_ALLOCATION", "0.2"))
    max_risk = float(os.getenv("MAX_RISK_PER_TOKEN", "0.1"))
    risk_mult = float(os.getenv("RISK_MULTIPLIER", "1.0"))
    amount = calculate_order_size(
        balance,
        1.0,
        0.0,
        0.0,
        risk_tolerance=rt * risk_mult,
        max_allocation=max_alloc * risk_mult,
        max_risk_per_token=max_risk * risk_mult,
        current_allocation=alloc,
    )
    if amount <= 0:
        return []
    return {"token": token, "side": "buy", "amount": amount, "price": 0.0}
