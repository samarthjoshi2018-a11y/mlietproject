from projectfinal.components.data_ingestion import DataIngestion
from projectfinal.components.data_validation import DataValidation
from projectfinal.components.data_transformation import DataTransformation
from projectfinal.components.model_trainer import ModelTrainer
import json
from projectfinal.exception.exception import MyException
from projectfinal.logging.logger import logging
from projectfinal.constants.training_pipeline.training_pipeline import TrainingPipeline
import sys
from projectfinal.constants.training_pipeline.batch_predection import SimplePredictor
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from projectfinal.entity.config_entity import TrainingPipelineConfig

app = Flask(__name__)
CORS(app)
predictor = SimplePredictor()

# Add a root route for health checks
@app.route('/')
def home():
    return jsonify({"message": "ML Model API is running!", "status": "healthy"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

def run_training():
    try:
        print("🎯 Starting TRUE Per-User Model Training Pipeline...")
        training_pipeline_config = TrainingPipelineConfig()
        pipeline = TrainingPipeline(training_pipeline_config)
        artifact = pipeline.run_pipeline()
        print("✨ Training Pipeline Completed Successfully!")
        return artifact
    except Exception as e:
        print(f"💥 Pipeline Execution Failed: {str(e)}")
        sys.exit(1)
        
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.json
        user_id = data['user_id']
        features = data.get('features',{})
        
        result = predictor.predict(features, user_id)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    # This function should only run when testing locally
    print("✅ Predictor loaded!")
    
    # Test with User 8 (who normally has speed ~45)
    test_data = {
        'speed': 100,
        'frequency_crossing': 2,
        'zoneradius': 100,
        'hour': 23,
        'latitude': 39.9526, 
        'longitude': -75.1652
    }
    
    user_id = 15
    
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
    # Only run the test locally, not on Railway
    if os.environ.get('RAILWAY_ENVIRONMENT') is None:
        print("🚀 Running in local mode - executing test...")
        main()
    
    # Always start the Flask server
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port)