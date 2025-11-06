const { spawn } = require('child_process');

function callPythonPrediction() {
    const testData = {
        user_id: 'USER003',
        timestamp: '2025-08-20T02:17:18.456Z',
        temperature: 100,
        day_of_week: 4,
        time_of_the_day: 'morning',
        weather_conditions: 'snowy',
        frequency_crossing: 47
    };

    const pythonProcess = spawn('python', ['main.py']);

    let resultData = "";

    pythonProcess.stdout.on('data', (data) => {
        resultData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error("Python Error:", data.toString());
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            try {
                const result = JSON.parse(resultData);
                
                if (result.error) {
                    console.error("❌ Prediction Error:", result.error);
                } else {
                    console.log("✅ Prediction Result:");
                    console.log(`   User: ${result.user_id}`);
                    console.log(`   Behavior: ${result.behavior}`);
                    console.log(`   Normal Probability: ${result.normal_prob}`);
                    console.log(`   Anomalous Probability: ${result.anomalous_prob}`);
                    console.log(`   Used User Model: ${result.used_user_model}`);
                    if (result.anomaly_reason) {
                        console.log(`   Reason: ${result.anomaly_reason}`);
                    }
                }
            } catch (err) {
                console.error("❌ Failed to parse JSON:", resultData);
            }
        } else {
            console.error("❌ Python process failed with code:", code);
        }
    });

    // Send JSON input to Python
    pythonProcess.stdin.write(JSON.stringify(testData));
    pythonProcess.stdin.end();
}

// Run the prediction
callPythonPrediction();