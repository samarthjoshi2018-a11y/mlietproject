import os 
import sys
from projectfinal.exception.exception import MyException
from projectfinal.logging.logger import logging
 
from projectfinal.components.data_ingestion import DataIngestion
from projectfinal.components.data_validation import DataValidation
from projectfinal.components.data_transformation import DataTransformation
from projectfinal.components.model_trainer import ModelTrainer


from projectfinal.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from projectfinal.entity.config_entity import TrainingPipelineConfig


from projectfinal.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact

class TrainingPipeline:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.training_pipeline_config=training_pipeline_config
        except Exception as e:
            raise MyException(e,sys)
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion_config=DataIngestionConfig(self.training_pipeline_config)
            data_ingestion=DataIngestion(data_ingestion_config)
            logging.info("Starting data ingestion")
            data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config=DataValidationConfig(self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
            logging.info("Starting data validation")
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info("Data validation completed")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config=DataTransformationConfig(self.training_pipeline_config)
            data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
            logging.info("Starting data transformation")
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")       
            return data_transformation_artifact
        except Exception as e:  
            raise MyException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config=ModelTrainerConfig(self.training_pipeline_config)
            model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=model_trainer_config)
            logging.info("Starting model trainer")
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            logging.info("Model trainer completed")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys)
        
        
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)        
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer_artifact
        
        
        except Exception as e:
            raise MyException(e,sys)