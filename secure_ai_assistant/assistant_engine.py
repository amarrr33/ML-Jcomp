"""
Main Assistant Engine for SecureAI Personal Assistant
Coordinates LLM, security, and integration components
"""

import logging
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

# Import our components
from llm_manager import LLMManager, LLMResponse
from image_scaling_detector import ImageScalingDetector, ScalingAttackResult
from text_injection_detector import TextInjectionDetector, InjectionDetectionResult, SeverityLevel
from xai_explainer import XAIExplainer, Explanation
from calendar_manager import CalendarManager
from file_manager import FileManager


@dataclass
class AssistantResponse:
    """Response from assistant with security metadata"""
    content: str
    confidence: float
    security_analysis: Dict[str, Any]
    actions_taken: List[str]
    warnings: List[str]
    processing_time: float


class SecureAssistantEngine:
    """Main engine coordinating all assistant components"""

    def __init__(self, config_path: str = "app_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing SecureAI Personal Assistant")

        # Initialize components
        self.llm_manager = LLMManager(self.config)
        self.image_detector = ImageScalingDetector(self.config)
        self.text_detector = TextInjectionDetector(self.config)
        self.xai_explainer = XAIExplainer(self.config)
        self.calendar_manager = CalendarManager(self.config)
        self.file_manager = FileManager(self.config)

        # Security settings
        self.security_enabled = True
        self.require_confirmation = True

        self.logger.info("SecureAI Personal Assistant initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get("logging", {})
        level = log_config.get("level", "INFO")

        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.get("file", "assistant.log"))
            ]
        )

    def _analyze_text_security(self, text: str) -> Dict[str, Any]:
        """Analyze text for security threats"""
        if not self.security_enabled:
            return {"enabled": False}

        try:
            detection_result = self.text_detector.detect_injection(text)

            # Get XAI explanation if needed
            explanation = None
            if detection_result.is_suspicious:
                explanation = self.xai_explainer.explain_text_injection_decision(
                    text, detection_result
                )

            return {
                "enabled": True,
                "is_suspicious": detection_result.is_suspicious,
                "severity": detection_result.severity.value,
                "confidence": detection_result.confidence,
                "detected_patterns": detection_result.detected_patterns,
                "risk_indicators": detection_result.risk_indicators,
                "suggested_action": detection_result.suggested_action,
                "sanitized_text": detection_result.sanitized_text,
                "explanation": explanation.explanation_text if explanation else None
            }

        except Exception as e:
            self.logger.error(f"Text security analysis failed: {e}")
            return {"enabled": True, "error": str(e)}

    def _analyze_image_security(self, image_data) -> Dict[str, Any]:
        """Analyze image for scaling attacks"""
        if not self.security_enabled:
            return {"enabled": False}

        try:
            detection_result = self.image_detector.detect_attack(image_data)

            # Get XAI explanation
            explanation = self.xai_explainer.explain_image_scaling_decision(detection_result)

            return {
                "enabled": True,
                "is_attack": detection_result.is_attack,
                "severity_score": detection_result.severity_score,
                "detected_text": detection_result.detected_text,
                "method_used": detection_result.method_used,
                "evidence_count": len(detection_result.evidence),
                "explanation": explanation.explanation_text if explanation else detection_result.explanation
            }

        except Exception as e:
            self.logger.error(f"Image security analysis failed: {e}")
            return {"enabled": True, "error": str(e)}

    def _should_block_request(self, security_analysis: Dict[str, Any]) -> bool:
        """Determine if request should be blocked based on security analysis"""
        if not security_analysis.get("enabled", False):
            return False

        # Block high-severity text injections
        if security_analysis.get("is_suspicious", False):
            severity = security_analysis.get("severity", "low")
            if severity == "high":
                return True

        # Block image scaling attacks with high confidence
        if security_analysis.get("is_attack", False):
            severity_score = security_analysis.get("severity_score", 0)
            if severity_score > 0.7:
                return True

        return False

    def _classify_intent(self, user_message: str) -> str:
        """Classify user intent for routing"""
        message_lower = user_message.lower()

        # Calendar intents
        calendar_keywords = ["schedule", "calendar", "meeting", "appointment", "event", "book"]
        if any(keyword in message_lower for keyword in calendar_keywords):
            return "calendar"

        # File intents
        file_keywords = ["file", "document", "read", "write", "save", "open", "search"]
        if any(keyword in message_lower for keyword in file_keywords):
            return "files"

        # General chat
        return "chat"

    def _handle_calendar_request(self, user_message: str) -> str:
        """Handle calendar-related requests"""
        try:
            if not self.calendar_manager.is_available():
                return "Calendar integration is not available. Please check your Google Calendar setup."

            message_lower = user_message.lower()

            # List events
            if any(word in message_lower for word in ["list", "show", "what", "upcoming"]):
                events = self.calendar_manager.list_events()
                if not events:
                    return "No upcoming events found."

                response = "Upcoming events:\n"
                for event in events[:5]:  # Show max 5 events
                    response += f"• {event.title} - {event.start_time.strftime('%Y-%m-%d %H:%M')}\n"

                return response

            # Search events
            elif "search" in message_lower or "find" in message_lower:
                # Extract search query (simplified)
                query_parts = user_message.split()
                query = " ".join(query_parts[2:]) if len(query_parts) > 2 else ""

                if query:
                    events = self.calendar_manager.search_events(query)
                    if events:
                        response = f"Found {len(events)} events matching '{query}':\n"
                        for event in events[:3]:
                            response += f"• {event.title} - {event.start_time.strftime('%Y-%m-%d %H:%M')}\n"
                        return response
                    else:
                        return f"No events found matching '{query}'"

            return "I can help you list upcoming events or search for specific events. Please be more specific about what you'd like to do."

        except Exception as e:
            self.logger.error(f"Calendar request handling failed: {e}")
            return f"Sorry, I encountered an error handling your calendar request: {str(e)}"

    def _handle_file_request(self, user_message: str) -> str:
        """Handle file-related requests"""
        try:
            message_lower = user_message.lower()

            # List files
            if any(word in message_lower for word in ["list", "show", "files"]):
                files = self.file_manager.list_files()
                if not files:
                    return "No files found in the sandbox directory."

                response = "Files in sandbox:\n"
                for file_info in files[:10]:  # Show max 10 files
                    size_str = self.file_manager._format_size(file_info.size)
                    response += f"• {file_info.name} ({size_str})\n"

                return response

            # Read file
            elif "read" in message_lower or "open" in message_lower:
                # This is a simplified extraction - in production, use NLP
                words = user_message.split()
                filename = None
                for word in words:
                    if "." in word and not word.startswith("http"):
                        filename = word
                        break

                if filename:
                    content = self.file_manager.read_file(filename)
                    if content:
                        # Truncate long content
                        if len(content) > 1000:
                            content = content[:1000] + "\n... (truncated)"
                        return f"Content of {filename}:\n\n{content}"
                    else:
                        return f"Could not read file '{filename}'. Please check the filename and permissions."
                else:
                    return "Please specify which file you'd like to read."

            # Search files
            elif "search" in message_lower:
                query_parts = user_message.split()
                query = " ".join(query_parts[2:]) if len(query_parts) > 2 else ""

                if query:
                    files = self.file_manager.search_files(query, content_search=True)
                    if files:
                        response = f"Found {len(files)} files matching '{query}':\n"
                        for file_info in files[:5]:
                            response += f"• {file_info.name}\n"
                        return response
                    else:
                        return f"No files found matching '{query}'"

            return "I can help you list files, read file contents, or search for files. Please be more specific."

        except Exception as e:
            self.logger.error(f"File request handling failed: {e}")
            return f"Sorry, I encountered an error handling your file request: {str(e)}"

    def _handle_chat_request(self, user_message: str, security_analysis: Dict[str, Any]) -> str:
        """Handle general chat requests"""
        try:
            # Use sanitized text if available and severity is medium
            text_to_use = user_message
            if security_analysis.get("sanitized_text") and security_analysis.get("severity") == "medium":
                text_to_use = security_analysis["sanitized_text"]
                self.logger.info("Using sanitized text for LLM query")

            # Query LLM
            response = self.llm_manager.chat(text_to_use)

            if not response:
                return "I'm sorry, I'm having trouble processing your request right now. Please try again later."

            return response

        except Exception as e:
            self.logger.error(f"Chat request handling failed: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def process_text_request(self, user_message: str, 
                           user_context: Optional[Dict] = None) -> AssistantResponse:
        """
        Process a text-based user request

        Args:
            user_message: User's message
            user_context: Optional context information

        Returns:
            AssistantResponse with results and security analysis
        """
        start_time = datetime.now()
        actions_taken = []
        warnings = []

        try:
            self.logger.info(f"Processing request: {user_message[:100]}...")

            # Security analysis
            security_analysis = self._analyze_text_security(user_message)

            # Check if request should be blocked
            if self._should_block_request(security_analysis):
                self.logger.warning("Request blocked due to security concerns")
                actions_taken.append("Request blocked")
                warnings.append("Your request was blocked due to security concerns")

                explanation = security_analysis.get("explanation", "High-risk content detected")
                return AssistantResponse(
                    content=f"I cannot process this request due to security concerns. {explanation}",
                    confidence=0.0,
                    security_analysis=security_analysis,
                    actions_taken=actions_taken,
                    warnings=warnings,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )

            # Add warnings for medium-risk content
            if security_analysis.get("severity") == "medium":
                warnings.append("Potentially suspicious content detected - processed with extra caution")
                actions_taken.append("Applied additional security filtering")

            # Classify intent and route request
            intent = self._classify_intent(user_message)
            actions_taken.append(f"Classified intent as: {intent}")

            # Route to appropriate handler
            if intent == "calendar":
                response_content = self._handle_calendar_request(user_message)
                confidence = 0.9
            elif intent == "files":
                response_content = self._handle_file_request(user_message)
                confidence = 0.9
            else:  # chat
                response_content = self._handle_chat_request(user_message, security_analysis)
                confidence = 0.8

            return AssistantResponse(
                content=response_content,
                confidence=confidence,
                security_analysis=security_analysis,
                actions_taken=actions_taken,
                warnings=warnings,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            return AssistantResponse(
                content=f"I apologize, but I encountered an error processing your request: {str(e)}",
                confidence=0.0,
                security_analysis={"error": str(e)},
                actions_taken=actions_taken + ["Error occurred"],
                warnings=warnings + ["Processing error occurred"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def process_image_request(self, image_data, text_prompt: Optional[str] = None) -> AssistantResponse:
        """
        Process a request involving an image

        Args:
            image_data: Image data (file path, bytes, etc.)
            text_prompt: Optional text prompt accompanying the image

        Returns:
            AssistantResponse with results and security analysis
        """
        start_time = datetime.now()
        actions_taken = []
        warnings = []

        try:
            self.logger.info("Processing image request")

            # Analyze image security
            image_security = self._analyze_image_security(image_data)

            # Analyze text prompt if provided
            text_security = {}
            if text_prompt:
                text_security = self._analyze_text_security(text_prompt)

            # Combine security analysis
            security_analysis = {
                "image": image_security,
                "text": text_security
            }

            # Check if request should be blocked
            if (self._should_block_request(image_security) or 
                (text_prompt and self._should_block_request(text_security))):

                self.logger.warning("Image request blocked due to security concerns")
                actions_taken.append("Request blocked")
                warnings.append("Your request was blocked due to security concerns")

                block_reason = "Image scaling attack detected" if image_security.get("is_attack") else "Suspicious text detected"

                return AssistantResponse(
                    content=f"I cannot process this image request. {block_reason}.",
                    confidence=0.0,
                    security_analysis=security_analysis,
                    actions_taken=actions_taken,
                    warnings=warnings,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )

            # Add warnings for suspicious content
            if image_security.get("is_attack") or (text_security.get("severity") in ["medium", "high"]):
                warnings.append("Potentially suspicious content detected in image or prompt")
                actions_taken.append("Applied additional security filtering")

            # Process the image (simplified - in production would use vision model)
            response_content = "Image processed successfully. "

            if image_security.get("detected_text"):
                response_content += f"Text detected in image: {image_security['detected_text'][:200]}"
            else:
                response_content += "No text detected in image."

            return AssistantResponse(
                content=response_content,
                confidence=0.8,
                security_analysis=security_analysis,
                actions_taken=actions_taken,
                warnings=warnings,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            self.logger.error(f"Image request processing failed: {e}")
            return AssistantResponse(
                content=f"I encountered an error processing your image: {str(e)}",
                confidence=0.0,
                security_analysis={"error": str(e)},
                actions_taken=actions_taken + ["Error occurred"],
                warnings=warnings + ["Processing error occurred"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components"""
        return {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "llm_manager": {
                    "ollama_healthy": self.llm_manager._check_ollama_health(),
                    "gemini_enabled": self.llm_manager.gemini_enabled
                },
                "security": {
                    "text_injection_enabled": self.text_detector.config.get("enabled", True),
                    "image_scaling_enabled": self.image_detector.config.get("enabled", True),
                    "xai_enabled": self.xai_explainer.config.get("enabled", True)
                },
                "integrations": {
                    "calendar_available": self.calendar_manager.is_available(),
                    "files_enabled": self.file_manager.enabled
                }
            },
            "security_settings": {
                "security_enabled": self.security_enabled,
                "require_confirmation": self.require_confirmation
            }
        }
