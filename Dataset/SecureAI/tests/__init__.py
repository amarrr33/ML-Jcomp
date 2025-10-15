"""
Test Suite for SecureAI - Stage 6
Comprehensive testing for all tools, agents, and pipeline
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all test modules
from test_detection_tools import TestDetectionTools
from test_alignment_tools import TestAlignmentTools
from test_xai_tools import TestXAITools
from test_learning_tools import TestLearningTools
from test_agents import TestAgents
from test_integration import TestIntegration

def run_all_tests():
    """Run complete test suite"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDetectionTools))
    suite.addTests(loader.loadTestsFromTestCase(TestAlignmentTools))
    suite.addTests(loader.loadTestsFromTestCase(TestXAITools))
    suite.addTests(loader.loadTestsFromTestCase(TestLearningTools))
    suite.addTests(loader.loadTestsFromTestCase(TestAgents))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*80)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
