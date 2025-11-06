const { spawn } = require('child_process');
const path = require('path');

function predictUserBehavior(testData) {
    return new Promise((resolve, reject) => {
        const inputData = JSON.stringify(testData);
        
        const pythonProcess = spawn('python', [path.join(__dirname, 'main.py')]);
        
        let result = '';
        let errorOutput = '';
        
        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            // Clean the result - take only the last line if there are multiple outputs
            const cleanResult = result.trim().split('\n').pop().trim();
            
            if (cleanResult) {
                try {
                    const parsedResult = JSON.parse(cleanResult);
                    resolve(parsedResult);
                } catch (parseError) {
                    reject(`Failed to parse JSON: ${parseError.message}\nRaw output: ${result}\nErrors: ${errorOutput}`);
                }
            } else {
                reject(`No output received. Python errors: ${errorOutput}`);
            }
        });
        
        pythonProcess.stdin.write(inputData);
        pythonProcess.stdin.end();
    });
}

// Test data
const testData = {
    user_id: 15,
    temprature: 100,
    frequency_crossing: 2,
    zoneradius: 100,
    hour: 23,
    latitude: 39.9526,
    longitude: -75.1652
};

// Execute prediction
predictUserBehavior(testData)
    .then(result => {
        if (result.error) {
            console.error('Error:', result.error);
        } else {
            console.log('=== PREDICTION RESULT ===');
            console.log(`User ID: ${result.user_id}`);
            console.log(`Behavior: ${result.behavior.toUpperCase()}`);
            console.log(`Normal Probability: ${result.normal_prob}`);
            console.log(`Anomalous Probability: ${result.anomalous_prob}`);
            console.log(`Used User Model: ${result.used_user_model}`);
            
            if (result.behavior === 'anomalous') {
                console.log(`Anomaly Reason: ${result.anomaly_reason}`);
            }
        }
    })
    .catch(error => {
        console.error('Prediction failed:', error);
    });