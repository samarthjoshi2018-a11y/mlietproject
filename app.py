from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from projectfinal.constants.training_pipeline.batch_predection import SimplePredictor

app = Flask(__name__)

# MongoDB connection
MONGO_DB_URL = os.getenv('MONGO_DB_URL', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_DB_URL)
db = client['users']
collection = db['rawdata']

predictor = SimplePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():

    try:
        user_id = request.form['user_id']
        
        # Fetch last entry for the user from MongoDB
        last_entry = collection.find_one(
            {'user_id': user_id},
            sort=[('timestamp', -1)]
        )
        
        if not last_entry:
            return render_template('error.html', error='User not found')
        
        # Check if behavior already exists and is not null/empty
        if last_entry.get('behavior') not in [None, '', 'null']:
            # Behavior already exists, return existing prediction
            existing_result = {
                'user_id': user_id,
                'behavior': last_entry['behavior'],
                'normal_prob': last_entry.get('normal_probability', 0.5),
                'anomalous_prob': last_entry.get('anomalous_probability', 0.5),
                'used_user_model': last_entry.get('used_user_model', False),
                'anomaly_reason': last_entry.get('anomaly_reason', ''),
                'feature_count': last_entry.get('feature_count', 0),
                'database_updated': False,  # No new update
                'document_id': str(last_entry['_id']),
                'from_existing': True  # Flag to indicate existing prediction
            }
            
            # Prepare data for display
            prediction_data = {
                'user_id': last_entry['user_id'],
                'timestamp': last_entry['timestamp'],
                'temperature': last_entry['temperature'],
                'day_of_week': last_entry['day_of_week'],
                'time_of_the_day': last_entry['time_of_the_day'],
                'weather_conditions': last_entry['weather_conditions'],
                'frequency_crossing': last_entry['frequency_crossing']
            }
            
            return render_template('result.html', 
                                 user_data=prediction_data,
                                 prediction_result=existing_result)
        
        # Prepare data for prediction (behavior is null/empty)
        prediction_data = {
            'user_id': last_entry['user_id'],
            'timestamp': last_entry['timestamp'],
            'temperature': last_entry['temperature'],
            'day_of_week': last_entry['day_of_week'],
            'time_of_the_day': last_entry['time_of_the_day'],
            'weather_conditions': last_entry['weather_conditions'],
            'frequency_crossing': last_entry['frequency_crossing']
        }
        
        # Use your prediction pipeline
        result = predictor.predict(prediction_data, user_id)
        
        # Update the original document with behavior prediction
        update_result = collection.update_one(
            {'_id': last_entry['_id']},
            {
                '$set': {
                    'behavior': result['behavior'],
                    'prediction_confidence': max(result['normal_prob'], result['anomalous_prob']),
                    'prediction_timestamp': datetime.now(),
                    'used_user_model': result['used_user_model'],
                    'anomaly_reason': result.get('anomaly_reason', ''),
                    'feature_count': result['feature_count'],
                    'normal_probability': result['normal_prob'],
                    'anomalous_probability': result['anomalous_prob']
                }
            }
        )
        
        # Verify the update was successful
        if update_result.modified_count == 0:
            print(f"Warning: No document was updated for user {user_id}")
        
        # Add database update info to result
        result['database_updated'] = update_result.modified_count > 0
        result['document_id'] = str(last_entry['_id'])
        result['from_existing'] = False  # Flag to indicate new prediction
        
        # Render result in a separate HTML template
        return render_template('result.html', 
                             user_data=prediction_data,
                             prediction_result=result)
        
    except Exception as e:
        return render_template('error.html', error=str(e))
    
    
    
if __name__ == '__main__':
    print("🚀 Starting Flask Server...")
    print("📊 Models loaded successfully!")
    print("🌐 Server running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)