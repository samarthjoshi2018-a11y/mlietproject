import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from projectfinal.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact
)
from projectfinal.entity.config_entity import ModelTrainerConfig
from projectfinal.exception.exception import MyException 
from projectfinal.logging.logger import logging
from projectfinal.utils.main_utils.utils import save_object, load_object

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.user_models = {}  # Store separate models for each user
            self.global_model = None  # Fallback model for new users
        except Exception as e:
            raise MyException(e, sys)
    
    def _get_classification_score(self, y_true, y_pred):
        """Calculate classification metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': report['1']['precision'] if '1' in report else 0,
            'recall': report['1']['recall'] if '1' in report else 0,
            'f1_score': report['1']['f1-score'] if '1' in report else 0,
            'confusion_matrix': cm.tolist()
        }
    
    def _train_user_specific_model(self, X_train, y_train, X_test, y_test, user_id):
        """Train a separate model for a specific user"""
        try:
            logging.info(f"Training model for User {user_id} with {len(X_train)} samples")
            
            # Check if we have both classes for meaningful training
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                logging.warning(f"User {user_id} has only one class ({unique_classes}), skipping user-specific model")
                return None, None
            
            # Use XGBoost for user-specific models
            model = XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate on test data
            if len(X_test) > 0:
                y_pred = model.predict(X_test)
                metrics = self._get_classification_score(y_test, y_pred)
            else:
                metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
            
            return model, metrics
            
        except Exception as e:
            logging.error(f"Failed to train model for User {user_id}: {str(e)}")
            return None, None
    
    def _train_global_model(self, X_train, y_train, X_test, y_test):
        """Train a global model for users with insufficient data"""
        try:
            logging.info(f"Training global model with {len(X_train)} samples")
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            metrics = self._get_classification_score(y_test, y_pred)
            
            return model, metrics
            
        except Exception as e:
            logging.error(f"Failed to train global model: {str(e)}")
            return None, None
    
    def _load_and_prepare_data(self):
        """Load data from .npy files and prepare for training"""
        try:
            # Load numpy arrays
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)
            
            # Convert to DataFrames
            train_df = pd.DataFrame(train_arr)
            test_df = pd.DataFrame(test_arr)
            
            # Load feature columns to properly name the DataFrame columns
            feature_columns_path = "final_model/feature_columns.pkl"
            if os.path.exists(feature_columns_path):
                feature_columns = load_object(feature_columns_path)
                # Last column is target, so features are all but last
                if len(feature_columns) == train_df.shape[1] - 1:
                    train_df.columns = feature_columns + ['behaviour']
                    test_df.columns = feature_columns + ['behaviour']
                    logging.info(f"Loaded data with columns: {train_df.columns.tolist()}")
            
            # Load user IDs
            user_ids_data = load_object(
                self.data_transformation_artifact.transformed_object_file_path.replace('.pkl', '_user_ids.pkl')
            )
            train_user_ids = user_ids_data['train_user_ids']
            test_user_ids = user_ids_data['test_user_ids']
            
            return train_df, test_df, train_user_ids, test_user_ids
            
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            raise MyException(e, sys)
    
    def _create_user_id_mapping(self, train_df, test_df, train_user_ids, test_user_ids):
        """Create proper mapping between data and user IDs"""
        try:
            # Since we lost userid during numpy conversion, we need to reconstruct
            # This assumes the order of rows is preserved between data and user IDs
            
            # Prepare features and target
            X_train = train_df.drop(columns=['behaviour'], axis=1)
            y_train = train_df['behaviour']
            X_test = test_df.drop(columns=['behaviour'], axis=1)
            y_test = test_df['behaviour']
            
            # Convert user IDs to numpy arrays for easy masking
            train_user_ids = np.array(train_user_ids)
            test_user_ids = np.array(test_user_ids)
            
            return X_train, y_train, X_test, y_test, train_user_ids, test_user_ids
            
        except Exception as e:
            logging.error(f"Failed to create user mapping: {str(e)}")
            raise MyException(e, sys)
    
    def train_true_per_user_models(self):
        """Main method to train true per-user models"""
        try:
            logging.info("Starting TRUE per-user model training")
            
            # STEP 1: Load data from .npy files
            train_df, test_df, train_user_ids, test_user_ids = self._load_and_prepare_data()
            
            # STEP 2: Create user ID mapping
            X_train, y_train, X_test, y_test, train_user_ids, test_user_ids = self._create_user_id_mapping(
                train_df, test_df, train_user_ids, test_user_ids
            )
            
            logging.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
            logging.info(f"Train users: {len(np.unique(train_user_ids))}, Test users: {len(np.unique(test_user_ids))}")
            
            # STEP 3: Identify users with sufficient data (>30 samples)
            user_counts = pd.Series(train_user_ids).value_counts()
            frequent_users = user_counts[user_counts > 30].index.tolist()
            
            logging.info(f"Found {len(frequent_users)} users with sufficient data for per-user modeling")
            logging.info(f"User distribution: {user_counts.describe()}")
            
            user_performance = {}
            trained_user_count = 0
            
            # STEP 4: Train separate models for each frequent user
            for user_id in frequent_users:
                # Get this user's data
                train_mask = (train_user_ids == user_id)
                test_mask = (test_user_ids == user_id)
                
                X_train_user = X_train[train_mask]
                y_train_user = y_train[train_mask]
                X_test_user = X_test[test_mask]
                y_test_user = y_test[test_mask]
                
                if len(X_train_user) > 0:
                    user_model, metrics = self._train_user_specific_model(
                        X_train_user, y_train_user, X_test_user, y_test_user, user_id
                    )
                    
                    if user_model is not None:
                        self.user_models[user_id] = user_model
                        user_performance[user_id] = metrics
                        trained_user_count += 1
                        
                        logging.info(f"✅ User {user_id}: {metrics['accuracy']:.3f} accuracy, "
                                   f"{len(X_train_user)} train samples, {len(X_test_user)} test samples")
                    else:
                        logging.warning(f"❌ Failed to train model for User {user_id}")
            
            # STEP 5: Train global model for remaining users
            remaining_train_mask = ~np.isin(train_user_ids, frequent_users)
            remaining_test_mask = ~np.isin(test_user_ids, frequent_users)
            
            if remaining_train_mask.sum() > 0:
                X_global_train = X_train[remaining_train_mask]
                y_global_train = y_train[remaining_train_mask]
                X_global_test = X_test[remaining_test_mask]
                y_global_test = y_test[remaining_test_mask]
                
                logging.info(f"Training global model with {len(X_global_train)} samples "
                           f"({len(np.unique(train_user_ids[remaining_train_mask]))} users)")
                
                self.global_model, global_metrics = self._train_global_model(
                    X_global_train, y_global_train, X_global_test, y_global_test
                )
                
                if self.global_model is not None:
                    logging.info(f"🌍 Global model: {global_metrics['accuracy']:.3f} accuracy, "
                               f"{global_metrics['precision']:.3f} precision, "
                               f"{global_metrics['recall']:.3f} recall")
                    user_performance['global'] = global_metrics
            else:
                logging.warning("No data remaining for global model training")
            
            # STEP 6: Save all models
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs("final_model", exist_ok=True)
            
            # Save user models
            for user_id, model in self.user_models.items():
                user_model_path = os.path.join(model_dir, f"user_{user_id}_model.pkl")
                save_object(user_model_path, model)
                
                # Also save to final_model for deployment
                final_user_path = os.path.join("final_model", f"user_{user_id}_model.pkl")
                save_object(final_user_path, model)
            
            # Save global model
            if self.global_model is not None:
                global_model_path = os.path.join(model_dir, "global_model.pkl")
                save_object(global_model_path, self.global_model)
                
                final_global_path = os.path.join("final_model", "global_model.pkl")
                save_object(final_global_path, self.global_model)
            
            # STEP 7: Save model registry
            model_registry = {
                'user_models': list(self.user_models.keys()),
                'has_global_model': self.global_model is not None,
                'user_performance': user_performance,
                'total_users_trained': trained_user_count,
                'frequent_users_count': len(frequent_users)
            }
            
            save_object(self.model_trainer_config.trained_model_file_path, model_registry)
            save_object(os.path.join("final_model", "model_registry.pkl"), model_registry)
            
            # STEP 8: Create Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=user_performance,
                test_metric_artifact=user_performance
                
            )
            
            logging.info(f"🎉 TRUE per-user modeling completed successfully!")
            logging.info(f"📊 Trained {trained_user_count} user models + {'global model' if self.global_model else 'no global model'}")
            logging.info(f"💾 Models saved to: {model_dir} and final_model/")
            
            return model_trainer_artifact
            
        except Exception as e:
            logging.error(f"TRUE per-user modeling failed: {str(e)}")
            raise MyException(e, sys)
    
    def initiate_model_trainer(self):
        try:
           model_trainer_artifact = self.train_true_per_user_models()
           return model_trainer_artifact
        except Exception as e:
           raise MyException(e, sys)
       
       
       