def interpolate_policy(old_probs, new_probs, alpha):
    """
    Smooth policy update inspired by diffusion models.
    """
    alpha = max(0.0, min(1.0, alpha))
    return {
        k: alpha * new_probs[k] + (1 - alpha) * old_probs[k]
        for k in old_probs
    }
