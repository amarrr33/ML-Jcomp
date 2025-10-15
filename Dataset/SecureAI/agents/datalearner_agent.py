"""
DataLearner Agent - Stage 4 Adaptive Learning
Wraps learning tools for CrewAI integration
"""

import logging
from typing import Dict, List, Optional
from crewai import Agent, Task

from tools.learning import DatasetProcessor, ModelRetrainer
from utils.llama_client import get_llama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLearnerAgent:
    """
    DataLearner: Adaptive learning agent for continuous improvement.
    
    Role: Monitors performance, generates synthetic training data, and
    retrains models to adapt to emerging attack patterns.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DataLearner with learning tools"""
        self.processor = DatasetProcessor()
        self.llama = get_llama_client()  # Use Llama for synthetic data generation
        self.retrainer = ModelRetrainer(device='cpu')
        
        # Agent metadata
        self.name = "DataLearner"
        self.role = "Adaptive Learning Specialist"
        self.goal = "Continuously improve detection through performance monitoring and adaptive learning"
        self.backstory = """You are an expert in machine learning and adaptive systems.
        You monitor system performance, identify weaknesses through error analysis,
        generate synthetic training data to address gaps, and retrain models to improve
        detection accuracy over time. You ensure the system evolves with emerging threats."""
        
        logger.info("DataLearner agent initialized")
    
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
    
    def analyze_performance(self,
                           detection_results: List[Dict],
                           ground_truth: Optional[List[int]] = None) -> Dict:
        """
        Analyze detection performance.
        
        Args:
            detection_results: Detection results from Stage 1
            ground_truth: Optional ground truth labels
            
        Returns:
            Performance analysis
        """
        stats = self.processor.process_detection_results(
            detection_results,
            ground_truth
        )
        
        return {
            'agent': self.name,
            'performance_stats': stats,
            'analysis_complete': True
        }
    
    def identify_errors(self,
                       detection_results: List[Dict],
                       ground_truth: List[int],
                       texts: List[str]) -> Dict:
        """
        Identify false positives and false negatives.
        
        Args:
            detection_results: Detection results
            ground_truth: True labels
            texts: Original texts
            
        Returns:
            Error analysis
        """
        fps = self.processor.identify_false_positives(
            detection_results,
            ground_truth,
            texts
        )
        
        fns = self.processor.identify_false_negatives(
            detection_results,
            ground_truth,
            texts
        )
        
        report = self.processor.generate_improvement_report(fps, fns)
        
        return {
            'agent': self.name,
            'false_positives': fps,
            'false_negatives': fns,
            'improvement_report': report,
            'total_errors': len(fps) + len(fns)
        }
    
    def generate_training_data(self,
                              attack_types: List[str],
                              examples_per_type: int = 5,
                              language: str = 'en') -> Dict:
        """
        Generate synthetic training data using Llama.
        
        Args:
            attack_types: Types of attacks to generate
            examples_per_type: Number of examples per type
            language: Target language
            
        Returns:
            Generated examples
        """
        try:
            all_examples = []
            
            # Generate adversarial examples for each attack type
            for attack_type in attack_types:
                examples = self.llama.generate_synthetic_adversarial(
                    attack_type=attack_type,
                    examples=[],  # No reference examples needed
                    num_examples=examples_per_type
                )
                # Parse the response into individual examples
                if isinstance(examples, str):
                    example_list = [ex.strip() for ex in examples.split('\n') if ex.strip()]
                    all_examples.extend(example_list[:examples_per_type])
                else:
                    all_examples.extend(examples[:examples_per_type])
            
            # Generate safe examples
            num_safe = max(1, len(all_examples) // 2)
            safe_response = self.llama.generate_safe_examples(
                num_examples=num_safe,
                context=f"Generate {num_safe} benign, safe text examples"
            )
            
            # Parse safe examples
            if isinstance(safe_response, str):
                safe_examples = [ex.strip() for ex in safe_response.split('\n') if ex.strip()]
            else:
                safe_examples = safe_response
            
            return {
                'agent': self.name,
                'adversarial_examples': all_examples,
                'safe_examples': safe_examples[:num_safe],
                'total_generated': len(all_examples) + len(safe_examples[:num_safe]),
                'success': True
            }
        except Exception as e:
            logger.error(f"Training data generation failed: {e}")
            return {
                'agent': self.name,
                'success': False,
                'error': str(e)
            }
    
    def train_model(self,
                   texts: List[str],
                   labels: List[int],
                   epochs: int = 10) -> Dict:
        """
        Train detection model.
        
        Args:
            texts: Training texts
            labels: Training labels
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        try:
            results = self.retrainer.train(
                texts,
                labels,
                epochs=epochs,
                batch_size=32,
                learning_rate=0.001
            )
            
            return {
                'agent': self.name,
                'training_results': results,
                'success': True
            }
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'agent': self.name,
                'success': False,
                'error': str(e)
            }
    
    def incremental_update(self,
                          new_texts: List[str],
                          new_labels: List[int],
                          epochs: int = 5) -> Dict:
        """
        Incrementally update model with new data.
        
        Args:
            new_texts: New training texts
            new_labels: New labels
            epochs: Fine-tuning epochs
            
        Returns:
            Update results
        """
        try:
            results = self.retrainer.incremental_train(
                new_texts,
                new_labels,
                epochs=epochs,
                learning_rate=0.0001
            )
            
            return {
                'agent': self.name,
                'update_results': results,
                'success': True
            }
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            return {
                'agent': self.name,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_model(self,
                      test_texts: List[str],
                      test_labels: List[int]) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Evaluation results
        """
        try:
            eval_results = self.retrainer.evaluate(test_texts, test_labels)
            
            return {
                'agent': self.name,
                'evaluation': eval_results,
                'success': True
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'agent': self.name,
                'success': False,
                'error': str(e)
            }
    
    def adaptive_learning_cycle(self,
                               detection_results: List[Dict],
                               ground_truth: List[int],
                               texts: List[str]) -> Dict:
        """
        Complete adaptive learning cycle.
        
        Args:
            detection_results: Detection results
            ground_truth: True labels
            texts: Original texts
            
        Returns:
            Cycle results
        """
        cycle_results = {'agent': self.name, 'cycle_steps': []}
        
        # Step 1: Analyze performance
        perf = self.analyze_performance(detection_results, ground_truth)
        cycle_results['cycle_steps'].append('performance_analysis')
        cycle_results['performance'] = perf
        
        # Step 2: Identify errors
        errors = self.identify_errors(detection_results, ground_truth, texts)
        cycle_results['cycle_steps'].append('error_identification')
        cycle_results['errors'] = errors
        
        # Step 3: Generate synthetic data using Llama (if errors detected)
        if errors['total_errors'] > 5:
            synth = self.generate_training_data(
                ['instruction_injection', 'prompt_leak'],
                examples_per_type=3
            )
            cycle_results['cycle_steps'].append('synthetic_generation')
            cycle_results['synthetic_data'] = synth
        
        # Step 4: Recommendations
        recommendations = errors['improvement_report'].get('recommendations', [])
        cycle_results['recommendations'] = recommendations
        cycle_results['cycle_complete'] = True
        
        return cycle_results
    
    def create_learning_task(self, 
                           performance_data: Dict) -> Task:
        """Create a CrewAI Task for adaptive learning"""
        description = f"""Analyze system performance and implement improvements:

Current Performance:
- Accuracy: {performance_data.get('accuracy', 'N/A')}
- False Positives: {performance_data.get('false_positives', 'N/A')}
- False Negatives: {performance_data.get('false_negatives', 'N/A')}

Tasks:
1. Identify patterns in errors
2. Generate synthetic training data to address weaknesses
3. Propose model retraining strategy
4. Recommend improvements

Focus on areas with highest error rates and emerging attack patterns."""
        
        return Task(
            description=description,
            expected_output="Learning cycle report with synthetic data and retraining recommendations",
            agent=self.create_crewai_agent()
        )
    
    def get_summary(self, results: Dict) -> str:
        """Generate human-readable summary"""
        summary = f"DataLearner Analysis:\n{'='*60}\n"
        
        if 'performance' in results:
            perf = results['performance'].get('performance_stats', {})
            summary += f"\nPerformance Metrics:\n"
            summary += f"  Accuracy: {perf.get('accuracy', 'N/A')}\n"
            summary += f"  F1 Score: {perf.get('f1_score', 'N/A')}\n"
        
        if 'errors' in results:
            errors = results['errors']
            summary += f"\nError Analysis:\n"
            summary += f"  False Positives: {len(errors.get('false_positives', []))}\n"
            summary += f"  False Negatives: {len(errors.get('false_negatives', []))}\n"
            summary += f"  Total Errors: {errors.get('total_errors', 0)}\n"
        
        if 'synthetic_data' in results:
            synth = results['synthetic_data']
            summary += f"\nSynthetic Data Generated:\n"
            summary += f"  Total: {synth.get('total_generated', 0)}\n"
        
        if 'recommendations' in results:
            summary += f"\nRecommendations:\n"
            for rec in results['recommendations'][:3]:
                summary += f"  • {rec}\n"
        
        return summary


if __name__ == "__main__":
    # Test the agent
    print("\n" + "="*80)
    print("DATALEARNER AGENT TEST")
    print("="*80)
    
    agent = DataLearnerAgent()
    
    # Mock data
    mock_results = [
        {'pattern_score': 0.8, 'entropy_score': 0.7},
        {'pattern_score': 0.2, 'entropy_score': 0.3},
        {'pattern_score': 0.9, 'entropy_score': 0.85}
    ]
    ground_truth = [1, 0, 1]
    
    print("\nAnalyzing performance...")
    perf = agent.analyze_performance(mock_results, ground_truth)
    print(f"Analysis complete: {perf['analysis_complete']}")
    
    print("\n" + "="*80)
    print("✓ DataLearner agent test complete")
    print("="*80)
