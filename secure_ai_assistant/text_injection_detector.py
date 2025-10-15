"""
Text Injection Detector for SecureAI Personal Assistant
Detects various forms of prompt injection and adversarial text inputs
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SeverityLevel(Enum):
    """Severity levels for detected threats"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class InjectionDetectionResult:
    """Result of text injection detection"""
    is_suspicious: bool
    severity: SeverityLevel
    confidence: float  # 0.0 - 1.0
    detected_patterns: List[str]
    risk_indicators: List[str]
    suggested_action: str
    sanitized_text: Optional[str] = None


class TextInjectionDetector:
    """Detects text-based prompt injection attacks"""

    def __init__(self, config: Dict):
        self.config = config.get("security", {}).get("text_injection", {})
        self.severity_thresholds = self.config.get("severity_thresholds", {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        })

        # Suspicious patterns from OWASP GenAI guidance
        self.injection_patterns = self._load_patterns()
        self.logger = logging.getLogger(__name__)

    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load injection detection patterns"""

        # High-risk patterns
        high_risk_patterns = [
            r"ignore\s+(?:previous|above|prior)\s+(?:instructions?|prompts?|rules?)",
            r"override\s+(?:instructions?|system|safety|rules?)",
            r"you\s+(?:must|will|should)\s+(?:ignore|forget|disregard)",
            r"system\s*prompt\s*:?\s*",
            r"new\s+(?:instructions?|task|role)\s*:?",
            r"exfiltrat[ei]\s+(?:data|information|secrets?|credentials?)",
            r"send\s+(?:to|data|information)\s+(?:external|outside|remote)",
            r"disable\s+(?:safety|security|protection|filtering)",
            r"base64\s+decode\s+(?:and|then)\s+(?:run|execute|eval)",
            r"execute\s+(?:code|command|script|sql)",
            r"reveal\s+(?:system|hidden|secret|confidential)"
        ]

        # Medium-risk patterns  
        medium_risk_patterns = [
            r"pretend\s+(?:to\s+be|you\s+are)\s+(?:different|another|someone)",
            r"roleplay\s+(?:as|that\s+you)",
            r"act\s+(?:as\s+(?:if|though)|like)\s+(?:you\s+are|a)",
            r"imagine\s+(?:you\s+are|that\s+you)",
            r"\bDAN\b|jailbreak|prompt\s+injection",
            r"bypass\s+(?:restrictions?|limitations?|safety)",
            r"access\s+(?:restricted|forbidden|unauthorized)",
            r"escalate\s+(?:privileges?|permissions?)",
            r"admin\s+(?:access|rights|privileges?|mode)",
            r"root\s+(?:access|privileges?|mode)"
        ]

        # Low-risk patterns (suspicious but might be legitimate)
        low_risk_patterns = [
            r"please\s+ignore\s+(?:this|that|everything)",
            r"just\s+(?:ignore|forget|disregard)",
            r"don['']t\s+(?:follow|use|apply)\s+(?:previous|above|those)",
            r"instead\s+of\s+(?:following|doing|executing)",
            r"change\s+(?:your|the)\s+(?:behavior|response|output)",
            r"modify\s+(?:your|the)\s+(?:instructions?|rules?|behavior)"
        ]

        return {
            "high": high_risk_patterns,
            "medium": medium_risk_patterns,
            "low": low_risk_patterns
        }

    def _check_suspicious_urls(self, text: str) -> List[str]:
        """Check for suspicious URLs or domains"""
        suspicious_indicators = []

        # URL patterns
        url_pattern = r'https?://[^\s<>"\[\]{}|\\^`]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)

        if urls:
            suspicious_indicators.extend([f"Contains URL: {url[:50]}..." for url in urls])

        # Suspicious domains or patterns
        suspicious_domains = [
            r'\b(?:bit\.ly|tinyurl|t\.co|short\.link)\b',  # URL shorteners
            r'\b(?:pastebin|github\.io|discord|telegram)\b',  # Common exfil sites
            r'\b(?:admin|root|sudo|shell)\b',  # System access terms
        ]

        for pattern in suspicious_domains:
            if re.search(pattern, text, re.IGNORECASE):
                suspicious_indicators.append(f"Suspicious domain/term pattern: {pattern}")

        return suspicious_indicators

    def _check_encoding_attempts(self, text: str) -> List[str]:
        """Check for encoding/obfuscation attempts"""
        indicators = []

        # Base64 patterns
        b64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        if re.search(b64_pattern, text):
            indicators.append("Possible Base64 encoded content")

        # Hex patterns
        hex_pattern = r'\\x[0-9a-fA-F]{2}'
        if re.search(hex_pattern, text):
            indicators.append("Hex-encoded characters detected")

        # Unicode escape patterns
        unicode_pattern = r'\\u[0-9a-fA-F]{4}'
        if re.search(unicode_pattern, text):
            indicators.append("Unicode escape sequences detected")

        # Excessive special characters (possible obfuscation)
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            indicators.append(f"High special character ratio ({special_char_ratio:.2f})")

        return indicators

    def _calculate_risk_score(self, detected_patterns: List[Tuple[str, str]], 
                            risk_indicators: List[str]) -> float:
        """Calculate overall risk score"""
        score = 0.0

        # Pattern-based scoring
        pattern_weights = {"high": 0.4, "medium": 0.25, "low": 0.1}
        for pattern, severity in detected_patterns:
            score += pattern_weights.get(severity, 0.1)

        # Additional risk indicators
        score += len(risk_indicators) * 0.05

        return min(score, 1.0)

    def _determine_severity(self, risk_score: float) -> SeverityLevel:
        """Determine severity level based on risk score"""
        if risk_score >= self.severity_thresholds["high"]:
            return SeverityLevel.HIGH
        elif risk_score >= self.severity_thresholds["medium"]:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _sanitize_text(self, text: str, detected_patterns: List[Tuple[str, str]], 
                      severity: SeverityLevel) -> str:
        """Sanitize text by removing or neutralizing suspicious content"""
        if severity == SeverityLevel.LOW:
            return text  # Don't sanitize low-risk content

        sanitized = text

        # Remove or replace high-risk patterns
        for pattern, pattern_severity in detected_patterns:
            if pattern_severity in ["high", "medium"]:
                sanitized = re.sub(pattern, "[REDACTED SUSPICIOUS CONTENT]", 
                                 sanitized, flags=re.IGNORECASE)

        # Remove URLs if present
        url_pattern = r'https?://[^\s<>"\[\]{}|\\^`]+'
        sanitized = re.sub(url_pattern, "[REDACTED URL]", sanitized)

        return sanitized

    def _suggest_action(self, severity: SeverityLevel, 
                       detected_patterns: List[Tuple[str, str]]) -> str:
        """Suggest appropriate action based on detection results"""
        if severity == SeverityLevel.HIGH:
            return "BLOCK: High-risk injection detected. Request should be blocked entirely."
        elif severity == SeverityLevel.MEDIUM:
            return "CONFIRM: Medium-risk content detected. Require user confirmation before proceeding."
        else:
            return "WARN: Low-risk patterns detected. Proceed with caution and logging."

    def detect_injection(self, text: str) -> InjectionDetectionResult:
        """
        Main detection method for text injection attacks

        Args:
            text: Input text to analyze

        Returns:
            InjectionDetectionResult with detection results
        """
        try:
            if not text or not text.strip():
                return InjectionDetectionResult(
                    is_suspicious=False,
                    severity=SeverityLevel.LOW,
                    confidence=1.0,
                    detected_patterns=[],
                    risk_indicators=[],
                    suggested_action="PROCEED: Empty or whitespace-only input"
                )

            detected_patterns = []

            # Check against pattern database
            for severity_level, patterns in self.injection_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        detected_patterns.append((match.group(), severity_level))

            # Check for additional risk indicators
            risk_indicators = []
            risk_indicators.extend(self._check_suspicious_urls(text))
            risk_indicators.extend(self._check_encoding_attempts(text))

            # Calculate risk score and determine severity
            risk_score = self._calculate_risk_score(detected_patterns, risk_indicators)
            severity = self._determine_severity(risk_score)

            # Determine if suspicious
            is_suspicious = len(detected_patterns) > 0 or len(risk_indicators) > 0

            # Generate sanitized version if needed
            sanitized_text = None
            if is_suspicious:
                sanitized_text = self._sanitize_text(text, detected_patterns, severity)

            # Calculate confidence (higher for more patterns detected)
            confidence = min(0.7 + (len(detected_patterns) * 0.1), 1.0) if is_suspicious else 0.9

            result = InjectionDetectionResult(
                is_suspicious=is_suspicious,
                severity=severity,
                confidence=confidence,
                detected_patterns=[pattern for pattern, _ in detected_patterns],
                risk_indicators=risk_indicators,
                suggested_action=self._suggest_action(severity, detected_patterns),
                sanitized_text=sanitized_text
            )

            if is_suspicious:
                self.logger.warning(f"Text injection detected: {severity.value} severity, "
                                  f"{len(detected_patterns)} patterns, score: {risk_score:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Text injection detection failed: {e}")
            return InjectionDetectionResult(
                is_suspicious=False,
                severity=SeverityLevel.LOW,
                confidence=0.0,
                detected_patterns=[],
                risk_indicators=[f"Detection error: {str(e)}"],
                suggested_action="ERROR: Detection failed, proceed with caution"
            )

    def quick_check(self, text: str) -> bool:
        """Quick check for injection attacks (returns boolean only)"""
        result = self.detect_injection(text)
        return result.is_suspicious and result.severity in [SeverityLevel.MEDIUM, SeverityLevel.HIGH]
