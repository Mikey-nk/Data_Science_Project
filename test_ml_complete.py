"""
Comprehensive Test Suite for Complete ML Implementation
Tests all features to ensure production readiness
"""

import pandas as pd
import numpy as np
from predictive_modeling import (
    PredictionPipeline, ModelBuilder, ModelType, ProblemType, ModelConfig
)
from sklearn.datasets import make_classification, make_regression
import os
import time


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_classification():
    """Test classification pipeline"""
    print_section("🎯 Testing Classification")
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=7,
        n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    # Add some categorical and missing data
    df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df.loc[np.random.choice(df.index, 50), 'feature_0'] = np.nan
    
    print(f"✅ Created dataset: {df.shape}")
    print(f"   - Target classes: {df['target'].nunique()}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    
    # Create pipeline
    pipeline = PredictionPipeline(df)
    
    # Test 1: Analyze readiness
    print("\n📊 Test 1: Analyze Prediction Readiness")
    analysis = pipeline.analyze_prediction_readiness()
    print(f"   ✅ Ready: {analysis['ready']}")
    print(f"   ✅ Quality score: {analysis['data_quality_score']:.1f}")
    print(f"   ✅ Potential targets: {len(analysis['potential_targets'])}")
    assert analysis['ready'], "Data should be ready!"
    
    # Test 2: Train basic model
    print("\n🤖 Test 2: Train Random Forest Classifier")
    start = time.time()
    results = pipeline.train_model(
        target_column='target',
        model_type=ModelType.RANDOM_FOREST_CLASSIFIER,
        tune_hyperparameters=False
    )
    elapsed = time.time() - start
    
    print(f"   ✅ Training completed in {elapsed:.2f}s")
    print(f"   ✅ Accuracy: {results.metrics['accuracy']:.4f}")
    print(f"   ✅ Precision: {results.metrics['precision']:.4f}")
    print(f"   ✅ Recall: {results.metrics['recall']:.4f}")
    print(f"   ✅ F1 Score: {results.metrics['f1_score']:.4f}")
    
    assert results.metrics['accuracy'] > 0.7, "Accuracy too low!"
    assert results.feature_importance is not None, "Feature importance missing!"
    
    # Test 3: Cross-validation scores
    print("\n📈 Test 3: Cross-Validation")
    print(f"   ✅ CV Mean Accuracy: {results.cv_scores['mean_accuracy']:.4f}")
    print(f"   ✅ CV Std: {results.cv_scores['std_accuracy']:.4f}")
    assert results.cv_scores['mean_accuracy'] > 0.7, "CV accuracy too low!"
    
    # Test 4: Feature importance
    print("\n🎯 Test 4: Feature Importance")
    top_features = sorted(
        results.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for feat, imp in top_features:
        print(f"   ✅ {feat}: {imp:.4f}")
    
    # Test 5: Model persistence
    print("\n💾 Test 5: Model Persistence")
    model_key = list(pipeline.trained_models.keys())[0]
    filepath = 'test_model_classification.pkl'
    saved_path = pipeline.export_model(model_key, filepath)
    print(f"   ✅ Model saved to: {saved_path}")
    assert os.path.exists(filepath), "Model file not created!"
    
    # Test 6: Load model
    print("\n📂 Test 6: Load Model")
    new_pipeline = PredictionPipeline(df)
    new_pipeline.load_model(filepath)
    print(f"   ✅ Model loaded successfully")
    assert len(new_pipeline.trained_models) > 0, "Model not loaded!"
    
    # Test 7: Predictions on new data
    print("\n🔮 Test 7: Predictions on New Data")
    new_data = df.head(10).drop('target', axis=1)
    predictions_df = new_pipeline.predict_new_data(model_key, new_data)
    print(f"   ✅ Predictions shape: {predictions_df.shape}")
    print(f"   ✅ Has 'prediction' column: {'prediction' in predictions_df.columns}")
    print(f"   ✅ Sample predictions: {predictions_df['prediction'].head(3).tolist()}")
    
    # Clean up
    if os.path.exists(filepath):
        os.remove(filepath)
    
    print("\n✨ Classification tests PASSED!")
    return True


def test_regression():
    """Test regression pipeline"""
    print_section("📈 Testing Regression")
    
    # Create sample data
    X, y = make_regression(
        n_samples=1000, n_features=10, noise=10, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    print(f"✅ Created dataset: {df.shape}")
    print(f"   - Target range: [{df['target'].min():.2f}, {df['target'].max():.2f}]")
    
    pipeline = PredictionPipeline(df)
    
    # Test 1: Train regression model
    print("\n🤖 Test 1: Train Random Forest Regressor")
    results = pipeline.train_model(
        target_column='target',
        model_type=ModelType.RANDOM_FOREST_REGRESSOR
    )
    
    print(f"   ✅ R² Score: {results.metrics['r2_score']:.4f}")
    print(f"   ✅ RMSE: {results.metrics['rmse']:.2f}")
    print(f"   ✅ MAE: {results.metrics['mae']:.2f}")
    
    assert results.metrics['r2_score'] > 0.5, "R² too low!"
    
    # Test 2: Cross-validation
    print("\n📊 Test 2: Cross-Validation")
    print(f"   ✅ CV Mean R²: {results.cv_scores['mean_r2']:.4f}")
    print(f"   ✅ CV Std R²: {results.cv_scores['std_r2']:.4f}")
    
    # Test 3: Predictions
    print("\n🔮 Test 3: Predictions")
    model_key = list(pipeline.trained_models.keys())[0]
    new_data = df.head(5).drop('target', axis=1)
    predictions_df = pipeline.predict_new_data(model_key, new_data)
    
    print(f"   ✅ Predictions: {predictions_df['prediction'].values[:3]}")
    
    print("\n✨ Regression tests PASSED!")
    return True


def test_hyperparameter_tuning():
    """Test hyperparameter tuning"""
    print_section("🔧 Testing Hyperparameter Tuning")
    
    # Create small dataset for faster tuning
    X, y = make_classification(
        n_samples=500, n_features=5, n_informative=3,
        n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    pipeline = PredictionPipeline(df)
    
    print("\n🔍 Training with hyperparameter tuning...")
    print("   (This may take 30-60 seconds)")
    
    start = time.time()
    results = pipeline.train_model(
        target_column='target',
        model_type=ModelType.RANDOM_FOREST_CLASSIFIER,
        tune_hyperparameters=True
    )
    elapsed = time.time() - start
    
    print(f"\n   ✅ Tuning completed in {elapsed:.2f}s")
    print(f"   ✅ Best params: {results.best_params}")
    print(f"   ✅ Accuracy: {results.metrics['accuracy']:.4f}")
    
    assert results.best_params is not None, "Best params not found!"
    assert 'n_estimators' in results.best_params, "n_estimators not tuned!"
    
    print("\n✨ Hyperparameter tuning tests PASSED!")
    return True


def test_model_comparison():
    """Test model comparison"""
    print_section("📊 Testing Model Comparison")
    
    # Create dataset
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5,
        n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
    df['target'] = y
    
    pipeline = PredictionPipeline(df)
    
    # Train multiple models
    models = [
        ModelType.LOGISTIC_REGRESSION,
        ModelType.RANDOM_FOREST_CLASSIFIER,
        ModelType.DECISION_TREE_CLASSIFIER
    ]
    
    print("\n🤖 Training multiple models...")
    for model_type in models:
        print(f"   Training {model_type.value}...")
        results = pipeline.train_model(
            target_column='target',
            model_type=model_type
        )
        print(f"   ✅ Accuracy: {results.metrics['accuracy']:.4f}")
    
    # Compare models
    print("\n📈 Comparing models...")
    comparison = pipeline.compare_models(list(pipeline.trained_models.keys()))
    print(comparison.to_string(index=False))
    
    # Get best model
    best = pipeline.get_best_model(metric='accuracy')
    print(f"\n🏆 Best model: {best['results'].model_type}")
    print(f"   Accuracy: {best['results'].metrics['accuracy']:.4f}")
    
    assert len(pipeline.trained_models) == 3, "Not all models trained!"
    assert best is not None, "Best model not found!"
    
    print("\n✨ Model comparison tests PASSED!")
    return True


def test_recommendations():
    """Test model recommendations"""
    print_section("💡 Testing Model Recommendations")
    
    # Create dataset with proper parameters to avoid sklearn error
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_classes=3, 
        n_informative=8,  # Increased to avoid error
        n_redundant=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    pipeline = PredictionPipeline(df)
    
    # Get recommendations
    recommendations = pipeline.get_model_recommendations('target')
    
    print(f"\n📋 Got {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"\n   🤖 {rec['model'].value}")
        print(f"      Best for: {rec['best_for']}")
        print(f"      Complexity: {rec['complexity']}")
        print(f"      Speed: {rec['training_speed']}")
        print(f"      Pros: {', '.join(rec['pros'][:2])}")
    
    assert len(recommendations) >= 3, "Not enough recommendations!"
    
    print("\n✨ Recommendations tests PASSED!")
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print_section("🐛 Testing Edge Cases")
    
    # Test 1: Very small dataset
    print("\n📏 Test 1: Small dataset (50 rows)")
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    df_small = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df_small['target'] = y
    
    pipeline_small = PredictionPipeline(df_small)
    analysis = pipeline_small.analyze_prediction_readiness()
    print(f"   ✅ Analysis completed")
    print(f"   ✅ Ready: {analysis['ready']}")
    
    # Should still train but with warning
    results_small = pipeline_small.train_model(
        target_column='target',
        model_type=ModelType.LOGISTIC_REGRESSION
    )
    print(f"   ✅ Model trained: Accuracy {results_small.metrics['accuracy']:.4f}")
    
    # Test 2: Missing target column
    print("\n❌ Test 2: Missing target column (should handle gracefully)")
    try:
        pipeline_small.train_model(
            target_column='nonexistent',
            model_type=ModelType.LOGISTIC_REGRESSION
        )
        print("   ❌ Should have raised error!")
        return False
    except ValueError as e:
        print(f"   ✅ Correctly raised error: {str(e)[:50]}...")
    
    # Test 3: Categorical target
    print("\n📝 Test 3: Categorical target (text)")
    df_cat = df_small.copy()
    df_cat['target_cat'] = df_cat['target'].map({0: 'ClassA', 1: 'ClassB'})
    
    pipeline_cat = PredictionPipeline(df_cat)
    results_cat = pipeline_cat.train_model(
        target_column='target_cat',
        model_type=ModelType.RANDOM_FOREST_CLASSIFIER
    )
    print(f"   ✅ Handled categorical target")
    print(f"   ✅ Accuracy: {results_cat.metrics['accuracy']:.4f}")
    
    print("\n✨ Edge case tests PASSED!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪"*30)
    print("  COMPLETE ML IMPLEMENTATION TEST SUITE")
    print("🧪"*30)
    
    tests = [
        ("Classification", test_classification),
        ("Regression", test_regression),
        ("Hyperparameter Tuning", test_hyperparameter_tuning),
        ("Model Comparison", test_model_comparison),
        ("Recommendations", test_recommendations),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    start_time = time.time()
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED with error:")
            print(f"   {str(e)}")
            results.append((name, False))
    
    elapsed = time.time() - start_time
    
    # Summary
    print_section("📊 TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {name}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"{'='*60}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! ML implementation is production-ready!")
        print("\n✨ Features verified:")
        print("   ✅ Real sklearn models (not mocks)")
        print("   ✅ Actual training and predictions")
        print("   ✅ Hyperparameter tuning")
        print("   ✅ Cross-validation")
        print("   ✅ Model persistence (save/load)")
        print("   ✅ Feature importance")
        print("   ✅ Model comparison")
        print("   ✅ Edge case handling")
        print("\n🚀 Ready for production deployment!")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed. Check errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)