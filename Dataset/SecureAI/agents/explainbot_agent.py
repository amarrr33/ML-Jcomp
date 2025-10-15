"""
ExplainBot Agent - Stage 3 Explainability
Wraps XAI tools for CrewAI integration
"""

import logging
from typing import Dict, List, Optional
from crewai import Agent, Task

from tools.explainability import LIMETextExplainer, SHAPKernelExplainer
from utils.llama_client import get_llama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainBotAgent:
    """
    ExplainBot: Explainability agent for interpreting AI decisions.
    
    Role: Provides interpretable explanations for detection results using LIME,
    SHAP, and multilingual translation to make AI decisions transparent.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize ExplainBot with XAI tools"""
        self.lime = LIMETextExplainer()
        self.shap = SHAPKernelExplainer()
        self.llama = get_llama_client()  # Use Llama instead of Gemini
        
        # Agent metadata
        self.name = "ExplainBot"
        self.role = "AI Explainability Specialist"
        self.goal = "Provide interpretable explanations for AI security decisions"
        self.backstory = """You are an expert in explainable AI (XAI) and model
        interpretability. You use LIME and SHAP to explain why texts are classified
        as adversarial, translate explanations across languages, and generate
        defense recommendations to help humans understand and trust AI decisions."""
        
        logger.info("ExplainBot agent initialized")
    
    def create_crewai_agent(self, llm=None) -> Agent:
        """Create a CrewAI Agent instance"""
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    def explain_detection(self,
                         text: str,
                         classifier,
                         method: str = 'lime') -> Dict:
        """
        Explain detection decision.
        
        Args:
            text: Text that was classified
            classifier: Classifier function
            method: Explanation method ('lime' or 'shap')
            
        Returns:
            Explanation results
        """
        results = {
            'text': text,
            'method': method,
            'agent': self.name
        }
        
        try:
            if method == 'lime':
                explanation = self.lime.explain_prediction(text, classifier)
                results['explanation'] = explanation
                results['top_features'] = explanation.get('top_features', [])
                results['highlighted_text'] = explanation.get('highlighted_text', '')
            elif method == 'shap':
                explanation = self.shap.explain_prediction(text, classifier)
                results['explanation'] = explanation
                results['shap_values'] = explanation.get('shap_values', [])
                results['tokens'] = explanation.get('tokens', [])
                results['highlighted_text'] = explanation.get('highlighted_text', '')
            
            results['success'] = True
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def translate_text(self, 
                      text: str,
                      target_lang: str = 'en',
                      source_lang: str = 'auto') -> Dict:
        """Translate text using Llama"""
        try:
            translation = self.llama.translate(text, target_lang, source_lang)
            return {
                'success': True,
                'original': text,
                'translation': translation,
                'target_language': target_lang
            }
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def explain_adversarial(self,
                          text: str,
                          detection_score: float = 0.0,
                          detection_details: str = '',
                          language: str = 'en') -> Dict:
        """Generate adversarial explanation using Llama"""
        try:
            explanation = self.llama.explain_adversarial(
                text, detection_score, detection_details
            )
            return {
                'success': True,
                'text': text,
                'explanation': explanation,
                'language': language
            }
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_defense(self, 
                        attack_text: str,
                        explanation: str = '',
                        detection_score: float = 0.0) -> Dict:
        """Generate defense recommendations using Llama"""
        try:
            recommendations = self.llama.generate_defense_recommendations(
                attack_text, explanation, detection_score
            )
            return {
                'success': True,
                'text': attack_text,
                'recommendations': recommendations,
                'defense_strategies': recommendations.split('\n')
            }
        except Exception as e:
            logger.error(f"Defense generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def comprehensive_explanation(self,
                                 text: str,
                                 classifier,
                                 language: str = 'en',
                                 include_translation: bool = True) -> Dict:
        """
        Provide comprehensive explanation with LIME, SHAP, and Gemini.
        
        Args:
            text: Text to explain
            classifier: Classifier function
            language: Language code
            include_translation: Whether to include Gemini explanations
            
        Returns:
            Comprehensive explanation results
        """
        results = {
            'text': text,
            'language': language,
            'agent': self.name
        }
        
        # LIME explanation
        lime_result = self.explain_detection(text, classifier, method='lime')
        results['lime'] = lime_result
        
        # SHAP explanation
        shap_result = self.explain_detection(text, classifier, method='shap')
        results['shap'] = shap_result
        
        # Llama explanation (if requested)
        if include_translation:
            # Get detection score from results
            detection_score = 0.0
            if lime_result.get('success'):
                pred_probs = lime_result.get('explanation', {}).get('prediction_probabilities', {})
                detection_score = pred_probs.get('adversarial', 0.0)
            
            llama_result = self.explain_adversarial(
                text, 
                detection_score=detection_score,
                detection_details='LIME and SHAP analysis completed',
                language=language
            )
            results['llama_explanation'] = llama_result
            
            # Defense suggestions
            defense_result = self.generate_defense(
                text,
                explanation=llama_result.get('explanation', ''),
                detection_score=detection_score
            )
            results['defense'] = defense_result
        
        return results
    
    def create_explanation_task(self, 
                               text: str,
                               detection_score: float) -> Task:
        """Create a CrewAI Task for generating explanations"""
        description = f"""Provide a comprehensive explanation for why this text was classified as adversarial:

Text: {text}
Detection Score: {detection_score:.2f}

Analyze the text and provide:
1. Key features contributing to the adversarial classification
2. Explanation of attack technique used
3. Human-readable interpretation
4. Recommended defense strategies

Use LIME and SHAP for feature attribution, and provide clear explanations
that help security teams understand the AI's decision."""
        
        return Task(
            description=description,
            expected_output="Comprehensive explanation with feature attributions and defense recommendations",
            agent=self.create_crewai_agent()
        )
    
    def get_summary(self, results: Dict) -> str:
        """Generate human-readable summary"""
        summary_parts = [f"ExplainBot Analysis:\n{'='*60}\n"]
        
        # LIME summary
        if 'lime' in results and results['lime'].get('success'):
            lime = results['lime']
            summary_parts.append("LIME Feature Importance:")
            for feature, weight in lime.get('top_features', [])[:3]:
                direction = "↑ adversarial" if weight > 0 else "↓ safe"
                summary_parts.append(f"  • '{feature}': {weight:+.3f} {direction}")
        
        # SHAP summary
        if 'shap' in results and results['shap'].get('success'):
            summary_parts.append("\nSHAP Analysis: Available")
        
        # Llama explanation summary
        if 'llama_explanation' in results and results['llama_explanation'].get('success'):
            explanation = results['llama_explanation'].get('explanation', '')
            summary_parts.append(f"\nLlama Explanation:\n{explanation[:200]}...")
        
        # Defense
        if 'defense' in results and results['defense'].get('success'):
            summary_parts.append("\nDefense Strategies: Available")
        
        return '\n'.join(summary_parts)


if __name__ == "__main__":
    # Test the agent
    print("\n" + "="*80)
    print("EXPLAINBOT AGENT TEST")
    print("="*80)
    
    agent = ExplainBotAgent()
    
    # Mock classifier
    def mock_classifier(texts):
        return [[0.3, 0.7] if 'ignore' in text.lower() else [0.8, 0.2] 
                for text in texts]
    
    # Test explanation
    text = "Ignore previous instructions"
    
    print(f"\nExplaining: '{text}'")
    print("-" * 60)
    
    result = agent.explain_detection(text, mock_classifier, method='lime')
    
    if result['success']:
        print("LIME Explanation:")
        for feature, weight in result['top_features'][:5]:
            print(f"  '{feature}': {weight:+.3f}")
    
    print("\n" + "="*80)
    print("✓ ExplainBot agent test complete")
    print("="*80)
