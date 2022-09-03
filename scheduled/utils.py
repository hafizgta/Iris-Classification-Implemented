from pydantic import BaseModel
from typing import Optional, Literal, Union,List


flower_name= ['setosa', 'versicolor', 'virginica']

class IrisIdentification(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

class IrisPrediction(IrisIdentification):
    class_id:int
    class_name:str
    score:float

class DataInput(BaseModel):
    identification:List[IrisIdentification]


class Custom_exception(Exception):
    def __init__(self,status_code:int, error_detail: str,response:dict):
        self.error_detail = error_detail
        self.response=response
        self.status_code=status_code

def preprocess_input(input_raw):
    preprocessed_input=[]
    temp=[input_raw[key] for key in input_raw]
    return temp