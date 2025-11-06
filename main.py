import json
import sys
import os

# Add your project path to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from projectfinal.constants.training_pipeline.batch_predection import SimplePredictor

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
    main()