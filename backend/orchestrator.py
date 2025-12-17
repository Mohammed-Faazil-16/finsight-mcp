import numpy as np

from backend.regime.wasserstein import compute_regime_distance
from backend.regime.regime_state import classify_regime
from backend.policy.diffusion import interpolate_policy
from backend.protocol.decision_rules import apply_protocol
from backend.explainability.rationale import generate_explanation
from backend.mcp import ModelContextProtocol


class NeuroQuantMCP:
    """
    NeuroQuant MCP Orchestrator
    ---------------------------
    Coordinates:
    - Model inference
    - Regime detection (Wasserstein distance)
    - Diffusion-inspired policy smoothing
    - Context-aware protocol rules
    - Explainability layer
    """

    def __init__(self):
        self.model = ModelContextProtocol()
        self.prev_probs = {
            "buy": 0.33,
            "hold": 0.34,
            "sell": 0.33
        }

    def _simulate_return_window(self, length=60):
        """
        Simulate historical returns for regime detection.
        Keeps Streamlit demo offline & deterministic.
        """
        return np.random.normal(loc=0.001, scale=0.02, size=length)

    def run(self, features, context, return_window):
        """
        Full NeuroQuant MCP pipeline:
        Model → Regime Detection → Diffusion Policy → Protocol → Explanation
        """

        # -----------------------------
        # Defensive fallback
        # -----------------------------
        if return_window is None or len(return_window) < 5:
            return_window = self._simulate_return_window()

        prev_window = return_window[:-5]

        # -----------------------------
        # Regime detection (Wasserstein)
        # -----------------------------
        distance = compute_regime_distance(return_window, prev_window)
        regime = classify_regime(distance)

        # -----------------------------
        # Base model prediction
        # -----------------------------
        raw_probs = self.model.model_predict_proba(features)

        # -----------------------------
        # Diffusion-inspired smoothing
        # -----------------------------
        alpha = min(1.0, distance / 0.1)
        smoothed_probs = interpolate_policy(
            self.prev_probs,
            raw_probs,
            alpha
        )

        # -----------------------------
        # Protocol decision layer
        # -----------------------------
        decision = apply_protocol(
            smoothed_probs,
            context,
            regime
        )

        # -----------------------------
        # Explainability
        # -----------------------------
        explanation = generate_explanation(
            decision["action"],
            regime,
            context,
            alpha
        )

        # -----------------------------
        # Update internal state
        # -----------------------------
        self.prev_probs = smoothed_probs

        return {
            "regime": regime,
            "raw_probs": raw_probs,
            "smoothed_probs": smoothed_probs,
            "decision": decision,
            "explanation": explanation
        }
