# (paste mcp.py content provided below)
# backend/mcp.py
import joblib
import numpy as np

class ModelContextProtocol:
    def __init__(self, model_path="models/invest_model_rf.joblib"):
        self.model = joblib.load(model_path)

    def model_predict_proba(self, features):
        """
        features: dict with keys momentum, volatility, pe_ratio, sector_signal, liquidity
        returns model class probabilities for [hold, buy, sell]
        """
        X = [[
            features.get("momentum", 0.0),
            features.get("volatility", 0.05),
            features.get("pe_ratio", 15.0),
            features.get("sector_signal", 0),
            features.get("liquidity", 0.5),
        ]]
        probs = self.model.predict_proba(X)[0]
        # sklearn returns columns in sorted label order; ensure order [0,1,2] (hold,buy,sell)
        return {"hold": float(probs[0]), "buy": float(probs[1]), "sell": float(probs[2])}

    def contextual_adjustment(self, probs, context):
        """
        context: dict with keys risk_tolerance ('low','medium','high'), position_exposure (0-1),
                 market_sentiment (-1..1), time_horizon ('short','medium','long')
        Apply simple, explainable rules to adjust probabilities.
        """
        adj = probs.copy()

        # Apply risk tolerance: high risk favors buy, low risk favors hold
        rt = context.get("risk_tolerance", "medium")
        if rt == "high":
            adj["buy"] += 0.05
            adj["hold"] -= 0.03
        elif rt == "low":
            adj["buy"] -= 0.05
            adj["hold"] += 0.03

        # Position exposure: if already high exposure, reduce buy probability
        exposure = context.get("position_exposure", 0.2)  # 0..1
        adj["buy"] *= max(0.2, 1.0 - exposure)  # scale down buys when exposure high

        # Market sentiment: push toward buy/sell
        sentiment = context.get("market_sentiment", 0.0)
        adj["buy"] += 0.06 * sentiment
        adj["sell"] -= 0.04 * sentiment

        # Time horizon: short-term reduces weight of momentum if horizon is long
        if context.get("time_horizon", "medium") == "long":
            # de-emphasize momentum-based buy
            adj["buy"] -= 0.02

        # normalize and prevent negatives
        for k in adj:
            adj[k] = max(adj[k], 0.0)
        s = sum(adj.values()) or 1.0
        for k in adj:
            adj[k] = adj[k] / s

        return adj

    def protocol_decision(self, adjusted_probs):
        """
        Map adjusted probabilities to discrete action and produce an explanation score and top reasons.
        """
        action = max(adjusted_probs.items(), key=lambda x: x[1])[0]  # 'buy'/'hold'/'sell'
        confidence = adjusted_probs[action]

        # Simple explainable reasons using relative probabilities
        reasons = []
        if adjusted_probs["buy"] > 0.5:
            reasons.append("Model + context strongly favors buying")
        if adjusted_probs["sell"] > 0.4:
            reasons.append("Sell signal is significant compared to alternatives")
        if adjusted_probs["hold"] > 0.4:
            reasons.append("Market conditions favor holding")

        # fallback reasons
        if not reasons:
            sorted_pairs = sorted(adjusted_probs.items(), key=lambda x: x[1], reverse=True)
            reasons.append(f"Top signal: {sorted_pairs[0][0]} (prob={sorted_pairs[0][1]:.2f})")

        return {
            "action": action,
            "confidence": round(float(confidence), 3),
            "reasons": reasons,
            "adjusted_probs": adjusted_probs
        }

    def run(self, features, context):
        """
        Full MCP pipeline: model -> context adjust -> protocol decision
        """
        probs = self.model_predict_proba(features)
        adjusted = self.contextual_adjustment(probs, context)
        decision = self.protocol_decision(adjusted)
        return {
            "features": features,
            "raw_model_probs": probs,
            "adjusted_probs": adjusted,
            "decision": decision
        }
