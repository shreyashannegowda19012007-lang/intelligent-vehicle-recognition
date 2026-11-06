"""
Hybrid Decision Engine for Intelligent Vehicle Recognition.
Compares classical CV and deep-learning predictions, detects taxonomy agreements/disagreements,
and executes configurable fusion and arbitration strategies.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.hybrid.config import DecisionStrategy, HybridConfig

logger = logging.getLogger(__name__)


class HybridDecisionEngine:
    """
    Modular decision engine comparing Classical CV and Deep-Learning predictions.
    Supports Strategies A (DL Primary), B (Agreement-Based), and C (Configurable Score Fusion).
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self.strategy = self.config.decision_strategy
        self.fusion_weights = self.config.fusion_weights
        self.uncertainty_threshold = self.config.uncertainty_threshold
        self.agreement_boost = self.config.agreement_confidence_boost

    def decide(
        self,
        vehicle_id: int,
        bbox: dict[str, int],
        detection_info: dict[str, Any],
        dl_pred: dict[str, Any],
        classical_pred: dict[str, Any],
        strategy_override: DecisionStrategy | None = None,
    ) -> dict[str, Any]:
        """
        Execute prediction comparison and decision fusion for a single detected vehicle.

        Args:
            vehicle_id: Sequential vehicle identifier.
            bbox: Bounding box coordinates {"x1": int, "y1": int, "x2": int, "y2": int}.
            detection_info: YOLO detection record {"class": str, "confidence": float}.
            dl_pred: Standardized deep learning prediction output.
            classical_pred: Standardized classical CV prediction output.
            strategy_override: Optional strategy override for ablation or testing.

        Returns:
            Standardized vehicle prediction schema matching project specifications.
        """
        t0 = time.perf_counter()
        current_strategy = strategy_override or self.strategy
        warnings: list[str] = []

        dl_status = dl_pred.get("status", "unavailable")
        cv_status = classical_pred.get("status", "unavailable")

        dl_avail = (dl_status == "success")
        cv_avail = (cv_status == "success")

        # 1. Determine Agreement across hierarchy
        agreement = False
        agreement_details = {
            "class_match": False,
            "make_match": False,
            "model_match": False,
            "year_match": False,
        }

        if dl_avail and cv_avail:
            # Match by class_id or normalized class_name
            dl_cid = dl_pred.get("class_id", -1)
            cv_cid = classical_pred.get("class_id", -2)

            if dl_cid >= 0 and cv_cid >= 0:
                class_match = (dl_cid == cv_cid)
            else:
                class_match = (
                    str(dl_pred.get("predicted_class", "")).strip().lower()
                    == str(classical_pred.get("predicted_class", "")).strip().lower()
                    and dl_pred.get("predicted_class") not in {"Unknown", ""}
                )

            make_match = (
                str(dl_pred.get("make", "")).strip().lower()
                == str(classical_pred.get("make", "")).strip().lower()
                and dl_pred.get("make") not in {"Unknown", ""}
            )
            model_match = (
                str(dl_pred.get("model", "")).strip().lower()
                == str(classical_pred.get("model", "")).strip().lower()
                and dl_pred.get("model") not in {"Unknown", ""}
            )
            year_match = (
                int(dl_pred.get("year", 0)) == int(classical_pred.get("year", 0))
                and int(dl_pred.get("year", 0)) > 0
            )

            agreement = bool(class_match)
            agreement_details = {
                "class_match": bool(class_match),
                "make_match": bool(make_match),
                "model_match": bool(model_match),
                "year_match": bool(year_match),
            }

            if not agreement:
                warnings.append(
                    f"Model disagreement: DL predicts '{dl_pred.get('predicted_class')}' "
                    f"while Classical predicts '{classical_pred.get('predicted_class')}'."
                )
        elif not cv_avail:
            warnings.append(
                f"Classical classifier unavailable ({classical_pred.get('warning', 'no checkpoint')}); "
                "relying solely on deep learning prediction."
            )
        elif not dl_avail:
            warnings.append(
                f"Deep learning classifier unavailable ({dl_pred.get('warning', 'no checkpoint')}); "
                "relying on classical CV fallback."
            )

        # 2. Apply Decision Strategy
        final_prediction: dict[str, Any] = {}
        applied_strategy = current_strategy.value if isinstance(current_strategy, DecisionStrategy) else str(current_strategy)

        if dl_avail and cv_avail:
            if current_strategy == DecisionStrategy.DEEP_LEARNING_PRIMARY:
                # Strategy A: DL is primary; classical is supporting signal
                final_prediction = {
                    "class_id": dl_pred.get("class_id", -1),
                    "predicted_class": dl_pred.get("predicted_class", "Unknown"),
                    "make": dl_pred.get("make", "Unknown"),
                    "model": dl_pred.get("model", "Unknown"),
                    "year": dl_pred.get("year", 0),
                    "confidence": dl_pred.get("confidence", 0.0),
                    "selected_source": "deep_learning",
                }

            elif current_strategy == DecisionStrategy.AGREEMENT_BASED:
                # Strategy B: Agreement-based confidence
                if agreement:
                    base_conf = float(dl_pred.get("confidence", 0.0))
                    # If classical classifier is calibrated, apply bounded boost
                    if classical_pred.get("is_calibrated", False):
                        hybrid_conf = min(1.0, round(base_conf + self.agreement_boost, 4))
                    else:
                        hybrid_conf = base_conf

                    final_prediction = {
                        "class_id": dl_pred.get("class_id", -1),
                        "predicted_class": dl_pred.get("predicted_class", "Unknown"),
                        "make": dl_pred.get("make", "Unknown"),
                        "model": dl_pred.get("model", "Unknown"),
                        "year": dl_pred.get("year", 0),
                        "confidence": hybrid_conf,
                        "selected_source": "consensus",
                    }
                else:
                    # Disagreement: Preserve individual predictions, do not invent weighted confidence
                    final_prediction = {
                        "class_id": dl_pred.get("class_id", -1),
                        "predicted_class": dl_pred.get("predicted_class", "Unknown"),
                        "make": dl_pred.get("make", "Unknown"),
                        "model": dl_pred.get("model", "Unknown"),
                        "year": dl_pred.get("year", 0),
                        "confidence": dl_pred.get("confidence", 0.0),
                        "selected_source": "deep_learning_fallback",
                    }

            elif current_strategy == DecisionStrategy.SCORE_FUSION:
                # Strategy C: Configurable score fusion
                w_dl = self.fusion_weights.get("deep_learning", 0.70)
                w_cv = self.fusion_weights.get("classical", 0.30)
                norm_w_dl = w_dl / (w_dl + w_cv)
                norm_w_cv = w_cv / (w_dl + w_cv)

                if classical_pred.get("is_calibrated", False):
                    # Both calibrated probabilities
                    s_dl = float(dl_pred.get("confidence", 0.0))
                    s_cv = float(classical_pred.get("confidence", 0.0))

                    if agreement:
                        # Both models predict the same class: valid linear opinion pool
                        fused_conf = round(norm_w_dl * s_dl + norm_w_cv * s_cv, 4)
                        primary_pred = dl_pred
                    else:
                        # Models disagree: select class candidate with higher weighted score
                        w_dl_score = norm_w_dl * s_dl
                        w_cv_score = norm_w_cv * s_cv
                        if w_dl_score >= w_cv_score:
                            primary_pred = dl_pred
                            denom = w_dl_score + w_cv_score
                            fused_conf = round(w_dl_score / denom, 4) if denom > 0 else round(s_dl, 4)
                        else:
                            primary_pred = classical_pred
                            denom = w_dl_score + w_cv_score
                            fused_conf = round(w_cv_score / denom, 4) if denom > 0 else round(s_cv, 4)

                    final_prediction = {
                        "class_id": primary_pred.get("class_id", -1),
                        "predicted_class": primary_pred.get("predicted_class", "Unknown"),
                        "make": primary_pred.get("make", "Unknown"),
                        "model": primary_pred.get("model", "Unknown"),
                        "year": primary_pred.get("year", 0),
                        "confidence": fused_conf,
                        "selected_source": "calibrated_score_fusion",
                    }
                else:
                    # Classical score is uncalibrated decision value; do not falsely treat as probability
                    warnings.append(
                        "Score fusion attempted with uncalibrated classical model; "
                        "falling back to deep-learning primary prediction to preserve calibrated probability."
                    )
                    final_prediction = {
                        "class_id": dl_pred.get("class_id", -1),
                        "predicted_class": dl_pred.get("predicted_class", "Unknown"),
                        "make": dl_pred.get("make", "Unknown"),
                        "model": dl_pred.get("model", "Unknown"),
                        "year": dl_pred.get("year", 0),
                        "confidence": dl_pred.get("confidence", 0.0),
                        "selected_source": "deep_learning",
                    }
            else:
                # Fallback to DL primary
                final_prediction = {
                    "class_id": dl_pred.get("class_id", -1),
                    "predicted_class": dl_pred.get("predicted_class", "Unknown"),
                    "make": dl_pred.get("make", "Unknown"),
                    "model": dl_pred.get("model", "Unknown"),
                    "year": dl_pred.get("year", 0),
                    "confidence": dl_pred.get("confidence", 0.0),
                    "selected_source": "deep_learning",
                }

        elif dl_avail:
            # DL only
            final_prediction = {
                "class_id": dl_pred.get("class_id", -1),
                "predicted_class": dl_pred.get("predicted_class", "Unknown"),
                "make": dl_pred.get("make", "Unknown"),
                "model": dl_pred.get("model", "Unknown"),
                "year": dl_pred.get("year", 0),
                "confidence": dl_pred.get("confidence", 0.0),
                "selected_source": "deep_learning_standalone",
            }
        elif cv_avail:
            # Classical only
            final_prediction = {
                "class_id": classical_pred.get("class_id", -1),
                "predicted_class": classical_pred.get("predicted_class", "Unknown"),
                "make": classical_pred.get("make", "Unknown"),
                "model": classical_pred.get("model", "Unknown"),
                "year": classical_pred.get("year", 0),
                "confidence": classical_pred.get("confidence", 0.0),
                "selected_source": "classical_cv_fallback",
            }
        else:
            # Neither available
            warnings.append("Both classification models failed or were unavailable.")
            final_prediction = {
                "class_id": -1,
                "predicted_class": "Unknown",
                "make": "Unknown",
                "model": "Unknown",
                "year": 0,
                "confidence": 0.0,
                "selected_source": "none",
            }

        # Check for uncertainty warnings
        if final_prediction.get("confidence", 0.0) < self.config.confidence_rejection_threshold:
            warnings.append(
                f"Prediction confidence ({final_prediction.get('confidence', 0.0):.2f}) "
                f"below rejection threshold ({self.config.confidence_rejection_threshold:.2f})."
            )

        # Append existing model warnings
        if dl_pred.get("warning"):
            warnings.append(f"[DL] {dl_pred['warning']}")
        if classical_pred.get("warning"):
            warnings.append(f"[Classical] {classical_pred['warning']}")

        decision_time_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # Assemble clean schema matching Prompt 6 specification
        return {
            "vehicle_id": vehicle_id,
            "bbox": bbox,
            "detection": {
                "class": detection_info.get("class_name", "car"),
                "confidence": round(float(detection_info.get("confidence", 0.0)), 4),
            },
            "classical_prediction": {
                "make": classical_pred.get("make", "Unknown"),
                "model": classical_pred.get("model", "Unknown"),
                "year": classical_pred.get("year", 0),
                "score": classical_pred.get("score", 0.0),
                "class_id": classical_pred.get("class_id", -1),
                "predicted_class": classical_pred.get("predicted_class", "Unknown"),
                "is_calibrated": classical_pred.get("is_calibrated", False),
                "latency_ms": classical_pred.get("inference_time_ms", 0.0),
                "status": cv_status,
            },
            "deep_learning_prediction": {
                "make": dl_pred.get("make", "Unknown"),
                "model": dl_pred.get("model", "Unknown"),
                "year": dl_pred.get("year", 0),
                "confidence": dl_pred.get("confidence", 0.0),
                "class_id": dl_pred.get("class_id", -1),
                "predicted_class": dl_pred.get("predicted_class", "Unknown"),
                "top_5": dl_pred.get("top_5", []),
                "latency_ms": dl_pred.get("inference_time_ms", 0.0),
                "status": dl_status,
            },
            "agreement": agreement,
            "agreement_details": agreement_details,
            "final_prediction": {
                "make": final_prediction.get("make", "Unknown"),
                "model": final_prediction.get("model", "Unknown"),
                "year": final_prediction.get("year", 0),
                "class_name": final_prediction.get("predicted_class", "Unknown"),
                "class_id": final_prediction.get("class_id", -1),
                "confidence": final_prediction.get("confidence", 0.0),
            },
            "decision_strategy": applied_strategy,
            "decision_latency_ms": decision_time_ms,
            "warnings": warnings,
        }
