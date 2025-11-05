import pandas as pd
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
    
    def _create_simple_user_features(self, data: dict, user_id: int):
        """Create simple user-aware features for explanation"""
        if user_id not in self.user_profiles:
            return data
            
        profile = self.user_profiles[user_id]
        result = data.copy()
        
        # Simple deviation calculation
        if 'speed_mean' in profile and 'speed_std' in profile:
            speed_dev = (data.get('speed', 0) - profile['speed_mean']) / max(profile['speed_std'], 1)
            result['speed_deviation'] = speed_dev
            
        if 'frequency_crossing_mean' in profile and 'frequency_crossing_std' in profile:
            cross_dev = (data.get('frequency_crossing', 0) - profile['frequency_crossing_mean']) / max(profile['frequency_crossing_std'], 1)
            result['crossing_deviation'] = cross_dev
            
        return result
    
    def _analyze_anomaly(self, model, input_data: dict, user_id: int):
        """Find which feature caused the anomaly"""
        if user_id not in self.user_profiles:
            return "Unusual behavior pattern detected"
            
        profile = self.user_profiles[user_id]
        reasons = []
        
        # Check each feature against user's normal pattern
        if 'speed' in input_data and 'speed_mean' in profile and 'speed_std' in profile:
            speed_dev = abs((input_data['speed'] - profile['speed_mean']) / max(profile['speed_std'], 1))
            if speed_dev > 2.0:
                reasons.append(f"speed ({input_data['speed']} vs normal {profile['speed_mean']})")
        
        if 'frequency_crossing' in input_data and 'frequency_crossing_mean' in profile:
            cross_dev = abs((input_data['frequency_crossing'] - profile['frequency_crossing_mean']) / max(profile.get('frequency_crossing_std', 1), 1))
            if cross_dev > 2.0:
                reasons.append(f"crossing frequency ({input_data['frequency_crossing']} vs normal {profile['frequency_crossing_mean']})")
        
        if 'hour' in input_data and 'hour_mean' in profile:
            hour_diff = abs(input_data['hour'] - profile['hour_mean'])
            if hour_diff > 4:
                reasons.append(f"unusual hour ({input_data['hour']} vs normal {profile['hour_mean']})")
        
        if reasons:
            return "Anomaly due to: " + ", ".join(reasons)
        else:
            return "Multiple unusual patterns"
    
    def predict(self, data: dict, user_id: int):
        # Create user-aware features for analysis
        data_with_features = self._create_simple_user_features(data, user_id)
        
        # Prepare data for model
        df = pd.DataFrame([data_with_features]).drop('userid', axis=1, errors='ignore')
        
        # Ensure correct columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        
        # Preprocess and predict
        processed = self.preprocessor.transform(df)
        model = self.models.get(user_id, self.global_model)
        
        pred = model.predict(processed)[0]
        proba = model.predict_proba(processed)[0]
        
        # Analyze anomaly reason
        anomaly_reason = ""
        if pred == 1:  # If anomalous
            anomaly_reason = self._analyze_anomaly(model, data, user_id)
        
        return {
            'user_id': user_id,
            'behavior': 'anomalous' if pred == 1 else 'normal',
            'normal_prob': float(proba[0]),
            'anomalous_prob': float(proba[1]),
            'used_user_model': user_id in self.models,
            'anomaly_reason': anomaly_reason
        }