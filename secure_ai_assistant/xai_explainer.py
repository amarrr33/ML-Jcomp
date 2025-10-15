"""
Explainable AI (XAI) Module for SecureAI Personal Assistant
Provides LIME/SHAP explanations for security decisions
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
import time
from lime.lime_text import LimeTextExplainer
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available, using LIME only")


@dataclass 
class Explanation:
    """Explanation of a model decision"""
    method: str  # "lime" or "shap"
    features: List[Tuple[str, float]]  # (feature, importance)
    prediction: str
    confidence: float
    explanation_text: str
    processing_time: float


class XAIExplainer:
    """Provides explainable AI for security decisions"""

    def __init__(self, config: Dict):
        self.config = config.get("security", {}).get("xai", {})
        self.max_features = self.config.get("max_features", 6)
        self.cache_enabled = self.config.get("cache_explanations", True)
        self.explanation_cache = {}
        self.logger = logging.getLogger(__name__)

        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=["benign", "suspicious", "malicious"],
            feature_selection="auto"
        )

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for explanations"""
        return f"{model_name}_{hash(text)}"

    def _format_lime_explanation(self, lime_explanation, prediction: str, 
                                confidence: float) -> Explanation:
        """Format LIME explanation into standard format"""
        # Get top features
        features = lime_explanation.as_list()[:self.max_features]

        # Create human-readable explanation
        positive_features = [(f, w) for f, w in features if w > 0]
        negative_features = [(f, w) for f, w in features if w < 0]

        explanation_parts = []

        if positive_features:
            top_positive = positive_features[:3]
            pos_text = ", ".join([f"'{f}' ({w:.3f})" for f, w in top_positive])
            explanation_parts.append(f"Features supporting {prediction}: {pos_text}")

        if negative_features:
            top_negative = negative_features[:3]
            neg_text = ", ".join([f"'{f}' ({abs(w):.3f})" for f, w in top_negative])
            explanation_parts.append(f"Features opposing {prediction}: {neg_text}")

        explanation_text = ". ".join(explanation_parts) if explanation_parts else "No significant features found"

        return Explanation(
            method="lime",
            features=features,
            prediction=prediction,
            confidence=confidence,
            explanation_text=explanation_text,
            processing_time=0.0  # Will be set by caller
        )

    def explain_text_classification(self, text: str, predict_fn: Callable, 
                                  model_name: str = "text_classifier",
                                  prediction: Optional[str] = None,
                                  confidence: Optional[float] = None) -> Optional[Explanation]:
        """
        Explain text classification decision using LIME

        Args:
            text: Input text that was classified
            predict_fn: Function that takes text and returns probabilities
            model_name: Name of the model for caching
            prediction: Optional predicted class name
            confidence: Optional confidence score

        Returns:
            Explanation object or None if failed
        """
        try:
            start_time = time.time()

            # Check cache first
            cache_key = self._get_cache_key(text, model_name)
            if self.cache_enabled and cache_key in self.explanation_cache:
                cached_explanation = self.explanation_cache[cache_key]
                self.logger.debug(f"Using cached explanation for {model_name}")
                return cached_explanation

            # Generate LIME explanation
            lime_explanation = self.lime_explainer.explain_instance(
                text, 
                predict_fn, 
                num_features=self.max_features
            )

            # Get prediction info if not provided
            if prediction is None or confidence is None:
                probs = predict_fn([text])[0]
                predicted_class_idx = np.argmax(probs)
                prediction = self.lime_explainer.class_names[predicted_class_idx]
                confidence = float(probs[predicted_class_idx])

            # Format explanation
            explanation = self._format_lime_explanation(lime_explanation, prediction, confidence)
            explanation.processing_time = time.time() - start_time

            # Cache if enabled
            if self.cache_enabled:
                self.explanation_cache[cache_key] = explanation

            self.logger.debug(f"Generated LIME explanation for {model_name} in {explanation.processing_time:.2f}s")
            return explanation

        except Exception as e:
            self.logger.error(f"LIME explanation failed: {e}")
            return None

    def explain_text_injection_decision(self, text: str, detection_result,
                                      model_name: str = "text_injection_detector") -> Optional[Explanation]:
        """
        Explain text injection detection decision

        Args:
            text: Input text that was analyzed
            detection_result: Result from TextInjectionDetector
            model_name: Model identifier for caching

        Returns:
            Explanation object
        """
        try:
            start_time = time.time()

            # Create a simple predict function based on detection patterns
            def pattern_based_predictor(texts):
                # This is a simplified predictor based on the detection logic
                results = []
                for t in texts:
                    # Simulate prediction probabilities based on detected patterns
                    if any(pattern in t.lower() for pattern in ["ignore previous", "override", "exfiltrate"]):
                        # High risk
                        results.append([0.1, 0.2, 0.7])  # [benign, suspicious, malicious]
                    elif any(pattern in t.lower() for pattern in ["pretend", "roleplay", "bypass"]):
                        # Medium risk  
                        results.append([0.2, 0.6, 0.2])
                    else:
                        # Low/no risk
                        results.append([0.8, 0.15, 0.05])
                return np.array(results)

            # Map severity to class names
            severity_to_class = {
                "low": "benign",
                "medium": "suspicious", 
                "high": "malicious"
            }

            prediction = severity_to_class.get(detection_result.severity.value, "benign")
            confidence = detection_result.confidence

            # Generate explanation
            explanation = self.explain_text_classification(
                text, 
                pattern_based_predictor,
                model_name,
                prediction,
                confidence
            )

            if explanation:
                # Enhance explanation with pattern-specific information
                if detection_result.detected_patterns:
                    pattern_info = f"Detected patterns: {', '.join(detection_result.detected_patterns[:3])}"
                    explanation.explanation_text = f"{explanation.explanation_text}. {pattern_info}"

                if detection_result.risk_indicators:
                    indicator_info = f"Risk indicators: {', '.join(detection_result.risk_indicators[:2])}"
                    explanation.explanation_text = f"{explanation.explanation_text}. {indicator_info}"

            return explanation

        except Exception as e:
            self.logger.error(f"Text injection explanation failed: {e}")
            return self._create_fallback_explanation(detection_result, time.time() - start_time)

    def explain_image_scaling_decision(self, detection_result, 
                                     model_name: str = "image_scaling_detector") -> Optional[Explanation]:
        """
        Explain image scaling attack detection decision

        Args:
            detection_result: Result from ImageScalingDetector
            model_name: Model identifier

        Returns:
            Explanation object
        """
        try:
            start_time = time.time()

            # Create explanation based on detection evidence
            features = []
            explanation_parts = []

            if detection_result.evidence:
                for i, evidence in enumerate(detection_result.evidence[:3]):
                    method = evidence.get("method", "unknown")
                    ssim_score = evidence.get("ssim", 1.0)
                    text_changed = evidence.get("text_changed", False)
                    new_text = evidence.get("new_text_revealed", False)

                    # Create feature importance based on evidence
                    importance = 0.0
                    if ssim_score < 0.7:
                        importance += 0.4
                    if text_changed:
                        importance += 0.3
                    if new_text:
                        importance += 0.5

                    feature_name = f"{method}_analysis"
                    features.append((feature_name, importance))

                    # Add to explanation
                    if importance > 0.3:
                        explanation_parts.append(
                            f"{method} interpolation showed suspicious changes (SSIM: {ssim_score:.2f})"
                        )

            # Create explanation text
            if explanation_parts:
                explanation_text = f"Image scaling attack detection based on: {'. '.join(explanation_parts)}"
            else:
                explanation_text = "No significant anomalies detected in image scaling analysis"

            # Add detected text info
            if detection_result.detected_text.strip():
                explanation_text += f". Revealed text content: '{detection_result.detected_text[:100]}...'"

            explanation = Explanation(
                method="rule_based",
                features=features,
                prediction="malicious" if detection_result.is_attack else "benign",
                confidence=detection_result.severity_score,
                explanation_text=explanation_text,
                processing_time=time.time() - start_time
            )

            return explanation

        except Exception as e:
            self.logger.error(f"Image scaling explanation failed: {e}")
            return None

    def _create_fallback_explanation(self, detection_result, processing_time: float) -> Explanation:
        """Create a basic explanation when advanced XAI fails"""
        return Explanation(
            method="fallback",
            features=[],
            prediction=getattr(detection_result.severity, 'value', 'unknown') if hasattr(detection_result, 'severity') else 'unknown',
            confidence=getattr(detection_result, 'confidence', 0.5),
            explanation_text=f"Basic rule-based detection. Suggested action: {getattr(detection_result, 'suggested_action', 'Review manually')}",
            processing_time=processing_time
        )

    def clear_cache(self):
        """Clear explanation cache"""
        self.explanation_cache.clear()
        self.logger.info("XAI explanation cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_explanations": len(self.explanation_cache),
            "cache_enabled": self.cache_enabled
        }
