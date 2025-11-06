import json
import sys
import os
from projectfinal.constants.training_pipeline.training_pipeline import TrainingPipelineConfig
from projectfinal.constants.training_pipeline.training_pipeline import TrainingPipeline
from projectfinal.exception.exception import MyException


# Add your project path to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from projectfinal.constants.training_pipeline.batch_predection import SimplePredictor


def run_training():
    """
    Function to run only the training pipeline
    """
    try:
        print("🎯 Starting Training Pipeline...")
        
        # Initialize training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        
        # Create training pipeline instance
        training_pipeline = TrainingPipeline(training_pipeline_config)
        
        # Run the complete pipeline
        model_trainer_artifact = training_pipeline.run_pipeline()
        
        print("✨ Training Pipeline Completed Successfully!")
        print("📊 Generated Artifacts:")
        print(f"   - Model: {model_trainer_artifact.trained_model_file_path}")
        
        return model_trainer_artifact
        
    except Exception as e:
        print(f"💥 Training Pipeline Failed: {str(e)}")
        raise MyException(e, sys)


def main():
    try:
        # Read JSON data from stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            print(json.dumps({'error': 'No input data received'}))
            return
        
        test_data = json.loads(input_data)
        
        # Initialize predictor
        predictor = SimplePredictor()
        
        # Extract user_id and features
        user_id = test_data.get('user_id', 15)
        features = {k: v for k, v in test_data.items() if k != 'user_id'}
        
        # Get prediction
        result = predictor.predict(features, user_id)
        
        # Print ONLY JSON result to stdout
        print(json.dumps(result))
        
    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'Invalid JSON input: {str(e)}'}))
    except Exception as e:
        print(json.dumps({'error': str(e)}))

if __name__ == "__main__":
    run_training()