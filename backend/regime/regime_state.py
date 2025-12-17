def classify_regime(distance, low_threshold=0.02, high_threshold=0.08):
    """
    Classifies market regime based on Wasserstein distance.
    """
    if distance > high_threshold:
        return "volatile"
    elif distance < low_threshold:
        return "stable"
    else:
        return "transition"
