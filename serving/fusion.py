"""Fusion of the Graph and Voice risk scores -- the "TrustMesh" combination step.

The two scores are NOT calibrated against each other: the graph score is an XGBoost probability
trained with a class-imbalance weight, the voice score is a spoof probability from a different
model on a different dataset. No dataset links a transaction to a voice clip, so there is no joint
ground truth to fit weights or thresholds to. This is a plain weighted average with placeholder
weights and bands; the component scores always travel with the combined one.
"""
import math

LEVELS = ("low", "medium", "high")
COMBINED_MEDIUM = 0.30  # placeholder band edges, not fit to anything
COMBINED_HIGH = 0.60

CAVEAT = ("Fusion weights and bands are placeholders: the graph and voice scores are not calibrated "
          "against each other, and no labeled dataset links transactions to voice clips, so the "
          "combined score has no validated accuracy.")


def combined_verdict(score):
    if score >= COMBINED_HIGH:
        return "high"
    if score >= COMBINED_MEDIUM:
        return "medium"
    return "low"


def fuse(graph_score, voice_score, w_graph=0.5, w_voice=0.5, visual_score=None, w_visual=0.0):
    """Weighted average (weights are normalized, so they need not sum to 1).

    `visual_score` is an optional third slot for a future Visual Engine signal. No visual model exists;
    the slot is "not connected" (None) by default, and then the result is exactly the two-signal average.
    """
    scores = [("graph_score", graph_score), ("voice_score", voice_score)]
    if visual_score is not None:
        scores.append(("visual_score", visual_score))
    for name, value in scores:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if visual_score is None:
        w_visual = 0.0  # an empty slot carries no weight, whatever was passed
    # Non-finite weights (inf) would turn the average into NaN, which was then stored and broke the JSON response.
    if not all(math.isfinite(w) for w in (w_graph, w_voice, w_visual)):
        raise ValueError("fusion weights must be finite numbers")
    if w_graph < 0 or w_voice < 0 or w_visual < 0 or (w_graph + w_voice + w_visual) == 0:
        raise ValueError("fusion weights must be non-negative and not all zero")
    total = w_graph + w_voice + w_visual
    wg, wv = w_graph / total, w_voice / total
    score = wg * graph_score + wv * voice_score
    weights = {"graph": wg, "voice": wv}
    if visual_score is not None:
        weights["visual"] = w_visual / total
        score += weights["visual"] * visual_score
    return {"combined_score": score, "combined_verdict": combined_verdict(score),
            "weights": weights, "signals_used": list(weights), "caveat": CAVEAT}
