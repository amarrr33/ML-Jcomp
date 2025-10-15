# Create the core LLM manager module
llm_manager_code = '''"""
Local LLM Manager for SecureAI Personal Assistant
Handles communication with Ollama and Gemini fallback
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai


@dataclass
class LLMResponse:
    """Response from LLM with metadata"""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    latency: float = 0.0
    confidence: float = 1.0


class LLMManager:
    """Manages local and cloud LLM interactions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ollama_endpoint = config["llm"]["endpoint"]
        self.ollama_model = config["llm"]["model"]
        self.gemini_enabled = config.get("gemini", {}).get("enabled", False)
        
        if self.gemini_enabled:
            # Initialize Gemini (API key should be set as environment variable)
            # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            pass
        
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _check_ollama_health(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            return False
    
    def _query_ollama(self, messages: List[Dict], **kwargs) -> Optional[LLMResponse]:
        """Query local Ollama instance"""
        try:
            start_time = time.time()
            
            payload = {
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config["llm"].get("temperature", 0.7),
                    "num_predict": self.config["llm"].get("max_tokens", 2048)
                }
            }
            
            response = requests.post(
                f"{self.ollama_endpoint}/api/chat",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            latency = time.time() - start_time
            
            return LLMResponse(
                content=result["message"]["content"],
                model=self.ollama_model,
                provider="ollama",
                latency=latency,
                confidence=1.0  # Assume high confidence for local model
            )
            
        except Exception as e:
            self.logger.error(f"Ollama query failed: {e}")
            return None
    
    def _query_gemini(self, messages: List[Dict], **kwargs) -> Optional[LLMResponse]:
        """Query Gemini as fallback"""
        if not self.gemini_enabled:
            return None
        
        try:
            # Rate limiting check
            current_time = time.time()
            if current_time - self.last_request_time < 1:  # 1 second minimum between requests
                time.sleep(1)
            
            start_time = time.time()
            
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_prompt(messages)
            
            # Note: This is a placeholder - actual Gemini API integration would go here
            # model = genai.GenerativeModel('gemini-pro')
            # response = model.generate_content(prompt)
            
            latency = time.time() - start_time
            self.last_request_time = time.time()
            
            return LLMResponse(
                content="[Gemini fallback not implemented in this demo]",
                model="gemini-pro",
                provider="gemini",
                latency=latency,
                confidence=0.9
            )
            
        except Exception as e:
            self.logger.error(f"Gemini query failed: {e}")
            return None
    
    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to a single prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\\n\\n".join(prompt_parts)
    
    def query(self, messages: List[Dict], use_fallback: bool = True) -> Optional[LLMResponse]:
        """
        Query LLM with fallback logic
        
        Args:
            messages: List of chat messages
            use_fallback: Whether to use Gemini if Ollama fails
        
        Returns:
            LLMResponse or None if all methods fail
        """
        # Try Ollama first
        if self._check_ollama_health():
            response = self._query_ollama(messages)
            if response:
                self.logger.info(f"Ollama response: {len(response.content)} chars, {response.latency:.2f}s")
                return response
        
        # Fall back to Gemini if enabled and Ollama fails
        if use_fallback and self.gemini_enabled:
            self.logger.info("Falling back to Gemini")
            response = self._query_gemini(messages)
            if response:
                self.logger.info(f"Gemini response: {len(response.content)} chars, {response.latency:.2f}s")
                return response
        
        self.logger.error("All LLM providers failed")
        return None
    
    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Simple chat interface
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt
        
        Returns:
            Assistant's response or None if failed
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": self.config["llm"]["system_prompt"]})
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.query(messages)
        return response.content if response else None
'''

with open('llm_manager.py', 'w') as f:
    f.write(llm_manager_code)

print("✓ llm_manager.py created")