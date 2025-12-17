def apply_protocol(probabilities, context, regime):
    """
    Applies business & risk protocol on probabilities.
    """
    probs = probabilities.copy()

    risk = context.get("risk_tolerance", "medium")
    exposure = context.get("position_exposure", 0.2)

    if risk == "low":
        probs["buy"] *= 0.7
        probs["hold"] *= 1.2
    elif risk == "high":
        probs["buy"] *= 1.2

    if exposure > 0.6:
        probs["buy"] *= 0.5

    if regime == "volatile":
        probs["sell"] *= 1.3

    # normalize
    total = sum(probs.values()) or 1.0
    for k in probs:
        probs[k] /= total

    action = max(probs, key=probs.get)

    return {
        "action": action,
        "confidence": round(probs[action], 3),
        "final_probs": probs
    }
