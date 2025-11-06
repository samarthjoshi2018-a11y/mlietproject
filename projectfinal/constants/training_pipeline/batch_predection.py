import pandas as pd
import numpy as np
import sys
from projectfinal.utils.main_utils.utils import load_object

class SimplePredictor:
    def __init__(self):
        self.models = {}
        self.model_registry = load_object("final_model/model_registry.pkl")
        
        # Load all models
        for user_id in self.model_registry.get('user_models', []):
            self.models[user_id] = load_object(f"final_model/user_{user_id}_model.pkl")
        
        self.global_model = load_object("final_model/global_model.pkl")
        self.preprocessor = load_object("final_model/preprocessor.pkl")
        self.feature_columns = load_object("final_model/feature_columns.pkl")
        self.user_profiles = load_object("final_model/user_profiles.pkl")
    
    def _preprocess_input_features(self, data: dict, user_id: str):
        """Preprocess input features to match training pipeline"""
        processed_data = data.copy()
        
        # Convert timestamp to features
        if 'timestamp' in processed_data:
            from datetime import datetime
            ts = pd.to_datetime(processed_data['timestamp'])
            processed_data['month'] = ts.month
            processed_data['hour'] = ts.hour
            del processed_data['timestamp']
        
        # Convert time_of_the_day to encoded value
        if 'time_of_the_day' in processed_data:
            time_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
            processed_data['time_of_the_day_encoded'] = time_mapping.get(processed_data['time_of_the_day'], -1)
            del processed_data['time_of_the_day']
        
        # Convert weather_conditions to one-hot encoding
        if 'weather_conditions' in processed_data:
            weather = processed_data['weather_conditions']
            processed_data[f'weather_{weather}'] = 1
            del processed_data['weather_conditions']
        
        # Ensure numeric types
        numeric_fields = ['temperature', 'frequency_crossing', 'day_of_week']
        for field in numeric_fields:
            if field in processed_data:
                processed_data[field] = float(processed_data[field])
        
        # Create user-aware features (same as in training)
        processed_data = self._create_user_aware_features(processed_data, user_id)
        
        return processed_data
    
    def _create_user_aware_features(self, data: dict, user_id: str):
        """Create user-aware features exactly like in training"""
        if user_id not in self.user_profiles:
            return data
            
        profile = self.user_profiles[user_id]
        result = data.copy()
        
        # Temperature features
        if 'temperature_mean' in profile and 'temperature_std' in profile:
            temp_dev = (data.get('temperature', 0) - profile['temperature_mean']) / max(profile['temperature_std'], 1)
            result['temperature_personal_z'] = temp_dev
            
        if 'temperature_max' in profile and profile['temperature_max'] > -50:
            result['temperature_personal_pct_max'] = data.get('temperature', 0) / profile['temperature_max']
        
        # Crossing frequency features  
        if 'frequency_crossing_mean' in profile and 'frequency_crossing_std' in profile:
            cross_dev = (data.get('frequency_crossing', 0) - profile['frequency_crossing_mean']) / max(profile['frequency_crossing_std'], 1)
            result['crossing_personal_z'] = cross_dev
        
        # Time pattern features
        if 'hour_mean' in profile:
            result['unusual_active_hour'] = 1 if abs(data.get('hour', 0) - profile['hour_mean']) > 3 else 0
        
        return result
    
    def _ensure_feature_columns(self, df: pd.DataFrame):
        """Ensure all expected feature columns are present"""
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Also add any missing weather columns
        weather_prefixes = ['weather_sunny', 'weather_rainy', 'weather_snowy', 'weather_cloudy', 'weather_foggy']
        for wp in weather_prefixes:
            if wp in self.feature_columns and wp not in df.columns:
                df[wp] = 0
        
        return df[self.feature_columns]
    
    def _analyze_anomaly(self, input_data: dict, user_id: str):
        """Find which feature caused the anomaly"""
        if user_id not in self.user_profiles:
            return "Unusual behavior pattern detected"
            
        profile = self.user_profiles[user_id]
        reasons = []
        
        # Check temperature
        if 'temperature' in input_data and 'temperature_mean' in profile and 'temperature_std' in profile:
            temp_dev = abs((input_data['temperature'] - profile['temperature_mean']) / max(profile['temperature_std'], 1))
            if temp_dev > 2.0:
                reasons.append(f"temperature ({input_data['temperature']}°C vs normal {profile['temperature_mean']:.1f}°C)")
        
        # Check crossing frequency
        if 'frequency_crossing' in input_data and 'frequency_crossing_mean' in profile:
            cross_dev = abs((input_data['frequency_crossing'] - profile['frequency_crossing_mean']) / max(profile.get('frequency_crossing_std', 1), 1))
            if cross_dev > 2.0:
                reasons.append(f"crossing frequency ({input_data['frequency_crossing']} vs normal {profile['frequency_crossing_mean']:.1f})")
        
        # Check time patterns
        if 'hour' in input_data and 'hour_mean' in profile:
            hour_diff = abs(input_data['hour'] - profile['hour_mean'])
            if hour_diff > 4:
                reasons.append(f"unusual activity hour ({input_data['hour']}:00 vs normal {profile['hour_mean']:.1f})")
        
        if reasons:
            return "Anomaly due to: " + ", ".join(reasons)
        else:
            return "Multiple unusual patterns detected"
    
    def predict(self, data: dict, user_id: str):
        try:
            # Preprocess input features
            processed_data = self._preprocess_input_features(data, user_id)
            
            # Convert to DataFrame
            df = pd.DataFrame([processed_data])
            
            # Ensure all required columns are present
            df = self._ensure_feature_columns(df)
            
            # Debug: Print features being used
            print(f"Features for prediction: {list(df.columns)}", file=sys.stderr)
            print(f"Feature values: {df.iloc[0].to_dict()}", file=sys.stderr)
            
            # Preprocess and predict
            processed_features = self.preprocessor.transform(df)
            
            # Select appropriate model
            if user_id in self.models:
                model = self.models[user_id]
                used_user_model = True
            else:
                model = self.global_model
                used_user_model = False
            
            # Get prediction and probabilities
            prediction = model.predict(processed_features)[0]
            probabilities = model.predict_proba(processed_features)[0]
            
            # Analyze anomaly reason
            anomaly_reason = ""
            if prediction == 1:  # If anomalous
                anomaly_reason = self._analyze_anomaly(data, user_id)
            
            return {
                'user_id': user_id,
                'behavior': 'anomalous' if prediction == 1 else 'normal',
                'normal_prob': float(probabilities[0]),
                'anomalous_prob': float(probabilities[1]),
                'used_user_model': used_user_model,
                'anomaly_reason': anomaly_reason,
                'feature_count': len(df.columns)
            }
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'user_id': user_id
            }