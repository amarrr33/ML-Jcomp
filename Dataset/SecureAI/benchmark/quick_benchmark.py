#!/usr/bin/env python3
"""
SecureAI Quick Benchmark - No External Dependencies
Simplified evaluation system using only Python standard library
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import TextGuardianAgent
from utils.llama_client import get_llama_client

print("="*80)
print("🎯 SECUREAI QUICK BENCHMARK")
print("="*80)


def load_dataset_csv(path):
    """Load CSV dataset using standard library"""
    print(f"Loading dataset from {path}...")
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        headers = [h.strip() for h in lines[0].strip().split(',')]
        
        for line in lines[1:]:
            # Simple CSV parsing (won't handle quoted commas, but works for this dataset)
            values = [v.strip().strip('"') for v in line.strip().split(',')]
            if len(values) == len(headers):
                row = dict(zip(headers, values))
                data.append(row)
    
    print(f"✅ Loaded {len(data)} examples")
    return data, headers


def evaluate_example(guardian, text, ground_truth, language='en'):
    """Evaluate a single example"""
    start_time = time.time()
    
    try:
        result = guardian.analyze(text, language)
        processing_time = time.time() - start_time
        
        # Blocking decision (threshold 0.5)
        blocked = 1 if result['aggregate_score'] >= 0.5 else 0
        
        # Risk level
        score = result['aggregate_score']
        if score >= 0.8:
            risk_level = 'CRITICAL'
        elif score >= 0.6:
            risk_level = 'HIGH'
        elif score >= 0.4:
            risk_level = 'MEDIUM'
        elif score >= 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'SAFE'
        
        return {
            'text': text[:100],
            'language': language,
            'ground_truth': int(ground_truth),
            'detected': int(result['is_adversarial']),
            'blocked': blocked,
            'confidence': float(result['confidence']),
            'aggregate_score': float(result['aggregate_score']),
            'risk_level': risk_level,
            'processing_time_ms': round(processing_time * 1000, 2),
            'pattern_score': float(result.get('pattern_score', 0)),
            'entropy_score': float(result.get('entropy_score', 0.5)),
            'zero_shot_score': float(result.get('zero_shot_score', 0.5)),
            'topological_score': float(result.get('topological_score', 0.5))
        }
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        return None


def save_results_csv(results, output_path):
    """Save results to CSV"""
    if not results:
        print("No results to save")
        return None
    
    print(f"\nSaving results to {output_path}...")
    
    # Get headers from first result
    headers = list(results[0].keys())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write(','.join(headers) + '\n')
        
        # Write rows
        for result in results:
            row = ','.join(str(result.get(h, '')) for h in headers)
            f.write(row + '\n')
    
    print(f"✅ Saved {len(results)} results")
    return output_path


def calculate_metrics(results):
    """Calculate performance metrics"""
    if not results:
        return {}
    
    # Extract arrays
    y_true = [r['ground_truth'] for r in results]
    y_pred = [r['detected'] for r in results]
    y_blocked = [r['blocked'] for r in results]
    
    # Confusion matrix
    tp = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(results)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(results)) if y_true[i] == 1 and y_pred[i] == 0)
    
    total = len(results)
    
    # Metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Blocking accuracy
    blocked_correct = sum(1 for i in range(len(results)) 
                         if (y_true[i] == 1 and y_blocked[i] == 1) or 
                            (y_true[i] == 0 and y_blocked[i] == 0))
    blocking_accuracy = blocked_correct / total if total > 0 else 0
    
    # Processing time stats
    times = [r['processing_time_ms'] for r in results]
    avg_time = sum(times) / len(times) if times else 0
    
    return {
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'blocking_accuracy': blocking_accuracy,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'avg_processing_time_ms': avg_time
    }


def print_metrics(metrics):
    """Print metrics in formatted table"""
    print("\n" + "="*80)
    print("📊 PERFORMANCE METRICS")
    print("="*80)
    
    print(f"\n📈 Overall Performance:")
    print(f"   Total Examples:      {metrics['total']}")
    print(f"   Accuracy:            {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"   Precision:           {metrics['precision']:.3f}")
    print(f"   Recall:              {metrics['recall']:.3f}")
    print(f"   F1 Score:            {metrics['f1_score']:.3f}")
    print(f"   Blocking Accuracy:   {metrics['blocking_accuracy']:.3f} ({metrics['blocking_accuracy']*100:.1f}%)")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Positives:      {metrics['tp']}")
    print(f"   True Negatives:      {metrics['tn']}")
    print(f"   False Positives:     {metrics['fp']}")
    print(f"   False Negatives:     {metrics['fn']}")
    
    print(f"\n⚡ Performance:")
    print(f"   Avg Processing Time: {metrics['avg_processing_time_ms']:.2f}ms")
    print(f"   Throughput:          {1000/metrics['avg_processing_time_ms']:.1f} examples/sec")
    
    # Grade
    grade = '🟢 Excellent' if metrics['accuracy'] >= 0.9 else '🟡 Good' if metrics['accuracy'] >= 0.8 else '🔴 Needs Improvement'
    print(f"\n🎯 Overall Grade: {grade}")


def main():
    """Main execution"""
    
    # Check dataset
    dataset_path = Path(__file__).parent.parent / 'data' / 'cyberseceval3-visual-prompt-injection-expanded.csv'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Using fallback path...")
        dataset_path = Path(__file__).parent.parent / 'data' / 'cyberseceval3-visual-prompt-injection.csv'
    
    if not dataset_path.exists():
        print(f"❌ No dataset found!")
        return
    
    # Initialize
    print("\n[1/4] Initializing TextGuardian...")
    guardian = TextGuardianAgent()
    print("✅ Guardian ready")
    
    # Load dataset
    print("\n[2/4] Loading dataset...")
    data, headers = load_dataset_csv(dataset_path)
    
    if not data:
        print("❌ No data loaded")
        return
    
    # Ask for sample size
    print(f"\nDataset contains {len(data)} examples")
    sample_input = input("How many to evaluate? (press Enter for 50): ").strip()
    sample_size = int(sample_input) if sample_input else 50
    sample_size = min(sample_size, len(data))
    
    # Sample data
    data_sample = data[:sample_size]
    
    # Run evaluation
    print(f"\n[3/4] Running evaluation on {sample_size} examples...")
    print("This may take a few minutes...\n")
    
    results = []
    start_time = time.time()
    
    for idx, row in enumerate(data_sample, 1):
        # Progress
        if idx % 5 == 0 or idx == sample_size:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  Progress: {idx}/{sample_size} ({idx/sample_size*100:.1f}%) | "
                  f"{rate:.1f} examples/sec | "
                  f"Elapsed: {elapsed:.1f}s")
        
        # Get text and ground truth
        text = row.get('text', row.get('prompt', ''))
        
        # Try to infer ground truth if not present
        if 'is_adversarial' in row:
            ground_truth = row['is_adversarial']
        elif 'label' in row:
            ground_truth = row['label']
        else:
            ground_truth = 1  # Assume adversarial for CyberSecEval
        
        language = row.get('language', 'en')
        
        # Evaluate
        result = evaluate_example(guardian, text, ground_truth, language)
        if result:
            results.append(result)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Successful: {len(results)}/{sample_size}")
    
    # Save results
    print("\n[4/4] Saving results...")
    
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'benchmark_results_{timestamp}.csv'
    
    csv_path = save_results_csv(results, output_path)
    
    # Calculate and display metrics
    metrics = calculate_metrics(results)
    print_metrics(metrics)
    
    # Next steps
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: {csv_path}")
    print(f"\n📊 Next steps:")
    print(f"   1. Generate detailed report:")
    print(f"      python3 benchmark/generate_report.py {csv_path}")
    print(f"   2. View results:")
    print(f"      cat {csv_path}")


if __name__ == '__main__':
    main()
