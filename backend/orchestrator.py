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
    - Regime detection
    - Policy smoothing
    - Context-aware protocol rules
    - Explainability
    """

    def __init__(self):
        self.model = ModelContextProtocol()
        self.prev_probs = {"buy": 0.33, "hold": 0.34, "sell": 0.33}

    def _simulate_return_window(self, length=60):
        """
        Simulate historical returns for regime detection.
        Keeps Streamlit demo offline & deterministic.
        """
        return np.random.normal(loc=0.001, scale=0.02, size=length)

    def run(self, features, context):
        # --------------------------------------------------
        # 1. Simulate return window (demo-safe)
        # --------------------------------------------------
        return_window = self._simulate_return_window()
        prev_window = return_window[:-5]

        # --------------------------------------------------
        # 2. Regime detection
        # --------------------------------------------------
        distance = compute_regime_distance(return_window, prev_window)
        regime = classify_regime(distance)

        # --------------------------------------------------
        # 3. Raw model probabilities
        # --------------------------------------------------
        raw_probs = self.model.model_predict_proba(features)

        # --------------------------------------------------
        # 4. Policy diffusion (temporal smoothing)
        # --------------------------------------------------
        alpha = min(1.0, distance / 0.1)
        smooth_probs = interpolate_policy(self.prev_probs, raw_probs, alpha)

        # --------------------------------------------------
        # 5. Context + protocol decision
        # --------------------------------------------------
        decision = apply_protocol(
            smooth_probs,
            context,
            regime
        )

        # --------------------------------------------------
        # 6. Explainability
        # --------------------------------------------------
        explanation = generate_explanation(
            decision["action"],
            regime,
            context,
            alpha
        )

        self.prev_probs = smooth_probs

        # --------------------------------------------------
        # 7. Unified output (UI-friendly)
        # --------------------------------------------------
        return {
            "regime": regime,
            "raw_model_probs": raw_probs,
            "adjusted_probs": smooth_probs,
            "decision": {
                **decision,
                "reasons": explanation
            }
        }
