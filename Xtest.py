import requests 
import json
import pandas as pd 
from machine_learning import df_to_X_y

sample = [100, 'diesel', 'grey', 'estate', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no']

url = 'http://127.0.0.1:8000/prediction'

input_data_for_model = {
    'engine_power' : 100,
    'fuel' : 'diesel',
    'paint_color' : 'grey',
    'car_type' : 'estate',
    'private_parking_available' : 'no',
    'has_gps' : 'no',
    'has_air_conditioning' : 'yes',
    'automatic_car' :'yes',
    'has_getaround_connect' : 'yes',    
    'has_speed_regulator' : 'yes',
    'winter_tires' : 'no'
}

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.text) 