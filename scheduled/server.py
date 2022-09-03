
from fastapi import FastAPI,Body
from fastapi.middleware.cors import CORSMiddleware
import warnings
import uvicorn
import os
import logging
from utils import DataInput,IrisPrediction,flower_name,preprocess_input
from typing import List,Dict
from iris import predict_iris


app = FastAPI(
    title="iris classification API",
    # description="""Obtain best hyperparameter for the given task."""
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def ping():
    return{'msg':'acknowledged'}


@app.post('/api/v1/iris_classification/prediction')
def predict(data_input:DataInput):
    preprocessed_data=[]
    result=[]
    for individual_data in data_input.identification:
        prep=individual_data.dict()
        result.append(prep)
        preprocessed_data.append(preprocess_input(prep))
    
    prediction=predict_iris(preprocessed_data)

    for i in range(len(result)):
        result[i]["class_id"]=int(prediction[i][0])
        result[i]["class_name"]=flower_name[prediction[i][0]]
        result[i]["score"]=float(prediction[i][1])
    print(result)
    return{"result": result }


if __name__ == '__main__':
    uvicorn.run("server:app", port=int(os.getenv('PORT','5000')), host=os.getenv('HOST','0.0.0.0'), reload=True)