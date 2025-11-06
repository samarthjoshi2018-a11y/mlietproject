from projectfinal.components.data_ingestion import DataIngestion
from projectfinal.components.data_validation import DataValidation
from projectfinal.components.data_transformation import DataTransformation
from projectfinal.components.model_trainer import ModelTrainer
import json
from projectfinal.exception.exception import MyException
from projectfinal.logging.logger import logging
from projectfinal.constants.training_pipeline.training_pipeline import TrainingPipeline
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from projectfinal.entity.config_entity import TrainingPipelineConfig
def run_training():
    try:
        print("🎯 Starting TRUE Per-User Model Training Pipeline...")
        
        # Initialize configuration
        training_pipeline_config = TrainingPipelineConfig()
        
        # Create pipeline instance
        pipeline = TrainingPipeline(training_pipeline_config)
        
        # Run the complete pipeline
        artifact = pipeline.run_pipeline()
        
        print("✨ Training Pipeline Completed Successfully!")
        print("📊 Generated Artifacts:")
        print(f"   - Models: {artifact.trained_model_file_path}")
        print(f"   - Training Metrics: Available in artifacts")
        
        return artifact
        
    except Exception as e:
        print(f"💥 Pipeline Execution Failed: {str(e)}")
        sys.exit(1)
        
        
        

from projectfinal.constants.training_pipeline.batch_predection import SimplePredictor

def main():
    predictor = SimplePredictor()
    print("✅ Predictor loaded!")
    
    # Test with User 8 (who normally has speed ~45)
    test_data = {
        'speed': 100,           # ⬅️ CHANGE THIS: 45 (normal) vs 100 (anomalous)
        'frequency_crossing':2, # ⬅️ CHANGE THIS: 2 (normal) vs 15 (anomalous)
        'zoneradius': 100,
        'hour': 23,
        'latitude': 39.9526, 
        'longitude': -75.1652
    }
    
    user_id = 15 # ⬅️ User who normally has speed ~45
    
    result = predictor.predict(test_data, user_id)
    json_result = json.dumps(result, indent=2)
    print("\n📝 Prediction Result (JSON):")
    print(json_result)
    
    print('normal result')
    print(f"\n📊 RESULTS for User {user_id}:")
    print(f"   Behavior: {result['behavior'].upper()}")
    print(f"   Normal: {result['normal_prob']:.3f}, Anomalous: {result['anomalous_prob']:.3f}")
    
    if result['behavior'] == 'anomalous':
        print(f"   🚨 REASON: {result['anomaly_reason']}")
    
    print(f"   Used User Model: {result['used_user_model']}")
    return json_result

if __name__ == "__main__":
    main()