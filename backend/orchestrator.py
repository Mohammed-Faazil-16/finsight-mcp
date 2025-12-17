import numpy as np
from backend.regime.wasserstein import compute_regime_distance
from backend.regime.regime_state import classify_regime
from backend.policy.diffusion import interpolate_policy
from backend.protocol.decision_rules import apply_protocol
from backend.explainability.rationale import generate_explanation
from backend.mcp import ModelContextProtocol


class NeuroQuantMCP:
    def __init__(self):
        self.model = ModelContextProtocol()
        self.prev_probs = {"buy": 0.33, "hold": 0.34, "sell": 0.33}

    def run(self, features, context, return_window):
        # simulate previous window (for demo)
        prev_window = return_window[:-5]

        distance = compute_regime_distance(return_window, prev_window)
        regime = classify_regime(distance)

        raw_probs = self.model.model_predict_proba(features)

        alpha = min(1.0, distance / 0.1)

        smooth_probs = interpolate_policy(self.prev_probs, raw_probs, alpha)

        decision = apply_protocol(smooth_probs, context, regime)

        explanation = generate_explanation(
            decision["action"], regime, context, alpha
        )

        self.prev_probs = smooth_probs

        return {
            "regime": regime,
            "raw_probs": raw_probs,
            "smoothed_probs": smooth_probs,
            "decision": decision,
            "explanation": explanation
        }
