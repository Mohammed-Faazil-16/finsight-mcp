def generate_explanation(action, regime, context, alpha):
    return [
        f"Detected market regime: {regime}",
        f"Risk tolerance: {context.get('risk_tolerance')}",
        f"Policy smoothed using diffusion (alpha={alpha:.2f})",
        f"Final decision: {action}"
    ]
