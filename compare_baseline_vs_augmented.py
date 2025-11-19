"""
Compare Baseline CNN vs Augmented CNN
======================================

Evaluates both models side-by-side to measure impact of augmentation.

Expected results:
- Augmented model should be more robust to noisy/distorted queries
- Baseline model may perform better on clean queries
- Augmented model should generalize better to real-world conditions
"""

import numpy as np
import sqlite3
from test_audio_matching import test_single_query


def compare_models(num_tests=20):
    """
    Compare baseline vs augmented CNN models.
    
    Tests both on clean audio to see which learned better features.
    """
    
    print("=" * 80)
    print("BASELINE CNN vs AUGMENTED CNN COMPARISON")
    print("=" * 80)
    
    # Check which models are available
    conn = sqlite3.connect('songs.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT model_name FROM track_embeddings ORDER BY model_name")
    available_models = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"\n📊 Available models in database: {available_models}\n")
    
    models_to_compare = []
    
    # Check for baseline CNN
    if 'cnn_64' in available_models:
        models_to_compare.append(('cnn_64', 'Baseline CNN (no augmentation)'))
    
    # Check for augmented CNN
    if 'cnn_aug_64' in available_models:
        models_to_compare.append(('cnn_aug_64', 'Augmented CNN (with augmentation)'))
    
    if len(models_to_compare) == 0:
        print("❌ No CNN models found! Please run:")
        print("   1. python cnn_classifier.py")
        print("   2. python cnn_extract_embeddings.py")
        print("   3. python cnn_classifier_augmented.py")
        print("   4. python cnn_extract_embeddings.py track_classifier_augmented.keras cnn_aug_64")
        return
    
    if len(models_to_compare) == 1:
        print(f"⚠️  Only 1 CNN model found: {models_to_compare[0][0]}")
        print("To compare, train both models!")
        return
    
    print(f"🔬 Comparing {len(models_to_compare)} models with {num_tests} test queries each\n")
    
    # Test each model
    results = {}
    
    for model_name, description in models_to_compare:
        print("=" * 80)
        print(f"Testing: {description} ({model_name})")
        print("=" * 80)
        
        # Run evaluation
        try:
            metrics = test_single_query(
                model_name=model_name,
                num_tests=num_tests,
                snippet_duration=5.0,
                verbose=False  # Less output for comparison
            )
            results[model_name] = (description, metrics)
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            results[model_name] = (description, None)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<30} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Avg Rank':<12} {'Avg Sim':<12}")
    print("-" * 80)
    
    for model_name, (description, metrics) in results.items():
        if metrics is None:
            print(f"{description:<30} {'ERROR':<12}")
            continue
        
        top1 = metrics['top1_accuracy']
        top5 = metrics['top5_accuracy']
        avg_rank = metrics['average_rank']
        avg_sim = metrics['average_similarity']
        
        print(f"{description:<30} {top1:>10.1%}  {top5:>10.1%}  {avg_rank:>10.2f}  {avg_sim:>10.3f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Analyze results
    if all(metrics is not None for _, metrics in results.values()):
        baseline_acc = results.get('cnn_64', (None, {'top1_accuracy': 0}))[1]['top1_accuracy']
        aug_acc = results.get('cnn_aug_64', (None, {'top1_accuracy': 0}))[1]['top1_accuracy']
        
        if 'cnn_64' in results and 'cnn_aug_64' in results:
            diff = aug_acc - baseline_acc
            
            print(f"\n📈 Augmented model vs Baseline:")
            print(f"   Top-1 Accuracy: {aug_acc:.1%} vs {baseline_acc:.1%} ({diff:+.1%})")
            
            if diff > 0.05:
                print("\n✅ Augmentation HELPED significantly (+5% or more)")
                print("   Learned features are more robust and generalizable")
            elif diff > 0:
                print("\n✅ Augmentation HELPED slightly")
                print("   Small improvement in generalization")
            elif diff > -0.05:
                print("\n➖ Augmentation had MINIMAL IMPACT")
                print("   Models perform similarly")
            else:
                print("\n⚠️  Augmentation HURT performance (-5% or more)")
                print("   May need to tune augmentation strength")
    
    print("\n" + "=" * 80)
    
    # Save results
    import json
    with open('model_comparison_results.json', 'w') as f:
        # Convert to serializable format
        save_results = {
            model: {
                'description': desc,
                'metrics': metrics
            }
            for model, (desc, metrics) in results.items()
            if metrics is not None
        }
        json.dump(save_results, f, indent=2)
    
    print("✓ Results saved to model_comparison_results.json")


if __name__ == '__main__':
    import sys
    
    num_tests = 30 if len(sys.argv) <= 1 else int(sys.argv[1])
    compare_models(num_tests=num_tests)
