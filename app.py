from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import os
import uvicorn
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from InputData import Inputdata
from projectfinal.constants.training_pipeline.batch_predection import SimplePredictor

app = FastAPI()
predictor = SimplePredictor()


@app.get('/')
def index():
    return "Welcome to the Anomaly Detection API"

@app.post('/predict')
def predict_datapoint(data: Inputdata):
    # Replace .dict() with .model_dump() for Pydantic V2
    input_data = data.dict()    
    
    result = predictor.predict(input_data, data.user_id)
    
    return JSONResponse(content={
        "behavior": result["behavior"],
        "anomaly_reason": result["anomaly_reason"]
    })
    
    
    
        
if __name__=='__main__':
   uvicorn.run(app, host='127.0.0.1', port=8000)