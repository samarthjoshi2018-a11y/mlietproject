import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from projectfinal.constants.training_pipeline import TARGET_COLUMN
from sklearn.pipeline import Pipeline

from projectfinal.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from projectfinal.entity.config_entity import DataTransformationConfig
from projectfinal.exception.exception import MyException 
from projectfinal.logging.logger import logging
from projectfinal.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            self.preprocessor = None
            self.user_behavior_profiles = {}  # 🆕 Store user behavior patterns
        except Exception as e:
            raise MyException(e, sys)
    
    @staticmethod  # ✅ FIX: Only one @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            # ✅ FIX: Convert numeric columns explicitly
            df = pd.read_csv(file_path)
            
            # Convert numeric columns that might be strings
            numeric_columns = ['temperature', 'frequency_crossing', 'day_of_week']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Log conversion results
                    converted_count = df[col].notna().sum()
                    total_count = len(df[col])
                    logging.info(f"Converted {col}: {converted_count}/{total_count} values to numeric")
            
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def _calculate_user_behavior_profiles(self, df: pd.DataFrame) -> dict:
        """Calculate baseline behavior patterns for each user"""
        logging.info("Calculating user behavior profiles")
        
        user_profiles = {}
        
        # Calculate statistics for each user
        user_stats = df.groupby('user_id').agg({
            'temperature': ['mean', 'std', 'max', 'min'],
            'frequency_crossing': ['mean', 'std', 'max'],
            'hour': ['mean', 'std'],  # Typical active hours
            'time_of_the_day_encoded': lambda x: x.mode()[0] if len(x.mode()) > 0 else -1  # Most common time
        }).round(3)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        
        # Convert to dictionary
        for user_id in user_stats.index:
            user_profiles[user_id] = user_stats.loc[user_id].to_dict()
        
        logging.info(f"Calculated behavior profiles for {len(user_profiles)} users")
        return user_profiles
    
    def _create_user_aware_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Create features that capture personal behavior patterns"""
        logging.info(f"Creating user-aware features for {dataset_name}")
        
        df_processed = df.copy()
        
        # 🆕 Calculate user behavior profiles from training data only
        if dataset_name == "train":
            self.user_behavior_profiles = self._calculate_user_behavior_profiles(df_processed)
        
        # 🆕 Create user-aware features
        user_aware_features = []
        
        for user_id in df_processed['user_id'].unique():
            if user_id in self.user_behavior_profiles:
                profile = self.user_behavior_profiles[user_id]
                
                # Features for this user
                user_mask = df_processed['user_id'] == user_id
                
                # Deviation from personal norm (Z-score)
                if profile['temperature_std'] > 0:  # Avoid division by zero
                    df_processed.loc[user_mask, 'temperature_personal_z'] = (
                        (df_processed.loc[user_mask, 'temperature'] - profile['temperature_mean']) / profile['temperature_std']
                    )
                else:
                    df_processed.loc[user_mask, 'temperature_personal_z'] = 0
                
                # Percentage of personal maximum
                if profile['temperature_max'] > -50:
                    df_processed.loc[user_mask, 'temperature_personal_pct_max'] = (
                        df_processed.loc[user_mask, 'temperature'] / profile['temperature_max']
                    )
                
                # Crossing frequency deviation
                if profile['frequency_crossing_std'] > 0:
                    df_processed.loc[user_mask, 'crossing_personal_z'] = (
                        (df_processed.loc[user_mask, 'frequency_crossing'] - profile['frequency_crossing_mean']) / profile['frequency_crossing_std']
                    )
                
                # Time pattern deviation
                df_processed.loc[user_mask, 'unusual_active_hour'] = (
                    abs(df_processed.loc[user_mask, 'hour'] - profile['hour_mean']) > 3
                ).astype(int)
                
                user_aware_features.extend([
                    'temperature_personal_z', 'temperature_personal_pct_max', 
                    'crossing_personal_z', 'unusual_active_hour'
                ])
        
        # 🆕 For users not in training profiles (new users in test set), use global averages
        missing_users = set(df_processed['user_id'].unique()) - set(self.user_behavior_profiles.keys())
        if missing_users and dataset_name == "test":
            logging.warning(f"Found {len(missing_users)} users not in training profiles, using global averages")
            
            # Calculate global averages from training profiles
            global_temperature_mean = np.mean([p['temperature_mean'] for p in self.user_behavior_profiles.values()])
            global_temperature_std = np.mean([p['temperature_std'] for p in self.user_behavior_profiles.values()])
            global_temperature_max = np.max([p['temperature_max'] for p in self.user_behavior_profiles.values()])
            
            for user_id in missing_users:
                user_mask = df_processed['user_id'] == user_id
                
                df_processed.loc[user_mask, 'temperature_personal_z'] = (
                    (df_processed.loc[user_mask, 'temperature'] - global_temperature_mean) / global_temperature_std
                )
                df_processed.loc[user_mask, 'temperature_personal_pct_max'] = (
                    df_processed.loc[user_mask, 'temperature'] / global_temperature_max
                )
                df_processed.loc[user_mask, 'crossing_personal_z'] = 0  # Neutral value
                df_processed.loc[user_mask, 'unusual_active_hour'] = 0  # Assume normal
        
        # Fill any remaining NaN values in user-aware features
        for feature in user_aware_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna(0)
        
        logging.info(f"Created user-aware features: {user_aware_features}")
        return df_processed
    
    def _preprocess_categorical_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Preprocess categorical features for geo-fenced security data"""
        logging.info(f"Preprocessing categorical features for {dataset_name}")
        
        df_processed = df.copy()
        
        # ✅ FIX: Convert numeric columns first
        numeric_columns = ['temperature', 'frequency_crossing', 'day_of_week']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                logging.info(f"Converted {col} to numeric in preprocessing")
        
        # Convert timestamp to datetime and extract useful features
        if 'timestamp' in df_processed.columns:
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
            df_processed['month'] = df_processed['timestamp'].dt.month
            df_processed['hour'] = df_processed['timestamp'].dt.hour
            df_processed = df_processed.drop('timestamp', axis=1)
        
        # Handle time_of_the_day categorical variable
        if 'time_of_the_day' in df_processed.columns:
            time_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
            df_processed['time_of_the_day_encoded'] = df_processed['time_of_the_day'].map(time_mapping)
            df_processed['time_of_the_day_encoded'] = df_processed['time_of_the_day_encoded'].fillna(-1)
            df_processed = df_processed.drop('time_of_the_day', axis=1)
        
        # Handle weather_conditions with one-hot encoding
        if 'weather_conditions' in df_processed.columns:
            weather_dummies = pd.get_dummies(df_processed['weather_conditions'], prefix='weather')
            df_processed = pd.concat([df_processed, weather_dummies], axis=1)
            df_processed = df_processed.drop('weather_conditions', axis=1)
        
        # 🆕 Create user-aware features AFTER basic preprocessing
        df_processed = self._create_user_aware_features(df_processed, dataset_name)
        
        logging.info(f"After categorical processing - {dataset_name} columns: {df_processed.columns.tolist()}")
        return df_processed
    
    def _clean_target_column(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Clean and validate behavior target column"""
        logging.info(f"Cleaning target column for {dataset_name} dataset")
        
        # Convert behavior to numeric if it's string
        if df[TARGET_COLUMN].dtype == 'object':
            behavior_mapping = {'normal': 0, 'anomalous': 1}
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map(behavior_mapping)
            unmapped_count = df[TARGET_COLUMN].isna().sum()
            if unmapped_count > 0:
                logging.warning(f"Found {unmapped_count} unmapped behavior values in {dataset_name}")
        
        return df
    
    def _remove_invalid_target_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid behavior labels"""
        original_shape = df.shape[0]
        
        # Remove rows with missing or invalid behavior (only keep 0 and 1)
        valid_mask = (~df[TARGET_COLUMN].isna()) & (df[TARGET_COLUMN].isin([0, 1]))
        df_clean = df[valid_mask]
        
        removed_count = original_shape - df_clean.shape[0]
        if removed_count > 0:
            logging.warning(f"Removed {removed_count} rows with invalid behavior labels")
        
        return df_clean

    def _handle_feature_missing_values(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Handle missing values in features with security-aware strategy"""
        logging.info(f"Handling missing values for {dataset_name}")
        
        df_processed = df.copy()
        missing_before = df_processed.isnull().sum().sum()
        
        if missing_before == 0:
            logging.info(f"No missing values found in {dataset_name} features")
            return df_processed
        
        logging.info(f"Found {missing_before} missing values in {dataset_name}")
        
        # Strategy 2: For numeric features - use median (preserves distribution)
        numeric_columns = ['temperature', 'frequency_crossing', 'hour', 'month', 'day_of_week', 
                          'time_of_the_day_encoded', 'temperature_personal_z', 'temperature_personal_pct_max', 'crossing_personal_z']
        for col in numeric_columns:
            if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
                logging.info(f"Filled {col} missing values with median: {median_val}")
        
        missing_after = df_processed.isnull().sum().sum()
        logging.info(f"Missing values reduced from {missing_before} to {missing_after}")
        
        return df_processed
    
    def _create_preprocessor(self, df: pd.DataFrame) -> Pipeline:
        """Create preprocessing pipeline for scaling features"""
        logging.info("Creating preprocessing pipeline")
        try:
            # Identify numeric columns (exclude target and any non-numeric)
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column if present
            if TARGET_COLUMN in numeric_features:
                numeric_features.remove(TARGET_COLUMN)
            
            # Remove user_id if it was accidentally included
            if 'user_id' in numeric_features:
                numeric_features.remove('user_id')
            
            logging.info(f"Numerical features for scaling: {numeric_features}")
            
            # Create preprocessing pipeline
            preprocessor = Pipeline([
                ('scaler', StandardScaler())  # Standardize features
            ])
            
            return preprocessor
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Convert user_id to string to avoid type issues
            train_df['user_id'] = train_df['user_id'].astype(str)
            test_df['user_id'] = test_df['user_id'].astype(str)
            
            # Store user_ids as strings BEFORE preprocessing
            train_user_ids = train_df['user_id'].copy()
            test_user_ids = test_df['user_id'].copy()
            
            # STEP 1: Preprocess categorical features
            logging.info("Preprocessing categorical features")
            train_df = self._preprocess_categorical_features(train_df, "train")
            test_df = self._preprocess_categorical_features(test_df, "test")

            # STEP 2: Clean target column
            logging.info("Cleaning target column")
            train_df = self._clean_target_column(train_df, "train")
            test_df = self._clean_target_column(test_df, "test")

            # STEP 3: Remove invalid target rows
            logging.info("Removing invalid target rows")
            train_df = self._remove_invalid_target_rows(train_df)
            test_df = self._remove_invalid_target_rows(test_df)

            # STEP 4: Handle feature missing values manually
            logging.info("Handling feature missing values")
            train_df = self._handle_feature_missing_values(train_df, "train")
            test_df = self._handle_feature_missing_values(test_df, "test")

            # Log final dataset info
            logging.info(f"Final train shape: {train_df.shape}, test shape: {test_df.shape}")
        
            # Prepare features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # STEP 5: Create and fit preprocessor
            logging.info("Creating and fitting preprocessor")
            preprocessor = self._create_preprocessor(input_feature_train_df)
            
            # ✅ FIX: Get numeric features and fit on them
            numeric_features_for_fit = input_feature_train_df.select_dtypes(include=[np.number]).columns.tolist()
            if TARGET_COLUMN in numeric_features_for_fit:
                numeric_features_for_fit.remove(TARGET_COLUMN)
            if 'user_id' in numeric_features_for_fit:
                numeric_features_for_fit.remove('user_id')
            
            logging.info(f"Fitting preprocessor on: {numeric_features_for_fit}")
            
            # ✅ FIX: Pass DataFrame subset, not list
            preprocessor_obj = preprocessor.fit(input_feature_train_df[numeric_features_for_fit])

            # STEP 6: Transform features using preprocessor
            logging.info("Step 6: Transforming features using preprocessor")
            # ✅ FIX: Transform same numeric features
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df[numeric_features_for_fit])
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df[numeric_features_for_fit])
            
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
            
            logging.info("Saving preprocessor object")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)
            save_object("final_model/preprocessor.pkl", preprocessor_obj)        
        
            # Save user_ids separately for per-user model training
            save_object(
                self.data_transformation_config.transformed_object_file_path.replace('.pkl', '_user_ids.pkl'),
                {
                    'train_user_ids': train_user_ids.values,
                    'test_user_ids': test_user_ids.values
                }
            )
            
            # Save feature columns for reference
            feature_columns = numeric_features_for_fit  # Save only numeric features used in scaling
            save_object("final_model/feature_columns.pkl", feature_columns)
            logging.info(f"Saved feature columns: {feature_columns}")

            # 🆕 Save user behavior profiles for prediction
            save_object("final_model/user_profiles.pkl", self.user_behavior_profiles)
            logging.info(f"Saved behavior profiles for {len(self.user_behavior_profiles)} users")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info("Data transformation completed successfully with user-aware features")
            return data_transformation_artifact

        except Exception as e:
            logging.error(f"Data transformation failed: {str(e)}")
            raise MyException(e, sys)