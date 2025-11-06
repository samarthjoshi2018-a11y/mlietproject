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




if __name__ == "__main__":
    run_training()