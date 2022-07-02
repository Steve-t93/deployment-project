from operator import index
from xml.dom.minidom import Element
from jsonschema import draft4_format_checker
import uvicorn
import json
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
import joblib

description = """
Getaround was founded in 2009 by Sam Zaid, Jessica Scorpio, and Elliot Kroo. In May 2011, Getaround won the TechCrunch Disrupt New York competition. 
In 2012, Getaround began serving Portland, Oregon with the aid of a $1.725 million grant from the Federal Highway Administration.


## Machine-Learning

You can predict a daily rental price by giving some characteristics.

check out the documentation for more information on the endpoint
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Endpoint using a Machine learning for estimate daily rental price"
    }
]

app = FastAPI(
    title= "🚗 Getaround API",
    description=description,
    contact={
        "name": "Getaround API - by Steve T",
    },
    openapi_tags=tags_metadata

)

app = FastAPI(debug=True)

class ModelInput(BaseModel):
    engine_power : int
    fuel : object
    paint_color : object
    car_type : object
    private_parking_available : object
    has_gps : object
    has_air_conditioning : object
    automatic_car : object
    has_getaround_connect : object    
    has_speed_regulator : object
    winter_tires : object

# load preprocessor
preprocessor = joblib.load("model\preprocessor.pkl") 
# load model
getaround_model = joblib.load("model\linear_regression.pkl")

@app.post("/prediction", tags=["Machine-Learning"])
async def getaround_pred(input_parameters: ModelInput):
    """
    This is a prediction for one rental price. This Endpoint will return a dictionnary like this:

    {"prediction": PREDICTION_VALUE[0,1]}

    Values must be given as a dictionary, here is an exemple:

    engine_power : int
    fuel : object
    paint_color : object
    car_type : object
    private_parking_available : object
    has_gps : object
    has_air_conditioning : object
    automatic_car : object
    has_getaround_connect : object    
    has_speed_regulator : object
    winter_tires : object

    """
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    # Dictionnary as dataframe for the prediction
    df = pd.DataFrame(input_dictionary, index=[0])
    # Preprocessing + model 
    X = preprocessor.transform(df.values)
    prediction = getaround_model.predict(X)

    return {"prediction": prediction.tolist()[0]}


if __name__=="__main__":
    uvicorn.run(app)

