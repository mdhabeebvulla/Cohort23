import pickle
from fastapi import FastAPI
from pydantic import BaseModel
#load iris pickle file
with open('iris_model.pkl','rb') as f:
    model = pickle.load(f)

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

@app.get("/")
def read_root():
    return{'message':'welcome to fastapi'}
@app.post("/predict/")
def predict(data:IrisInput):
    input_data = [[data.sepal_length,data.sepal_width,data.petal_length,data.petal_width]]
    prediction = model.predict(input_data)[0]
    return {'prediction':int(prediction)}
