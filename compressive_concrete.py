import numpy as np
import pandas as pd
import pickle
from pyforest import *




def load_model(path):
    model = pickle.load(open(path,'rb'))
    return model

def predict_concrete_strength(model, featureDict={}):
    print('Predicting the value...')
    # print("Enter the following parameters for the concrete mix:")
    # cement = float(input("Cement (kg/m^3): "))
    # slag = float(input("Slag (kg/m^3): "))
    # ash = float(input("Fly ash (kg/m^3): "))
    # water = float(input("Water (kg/m^3): "))
    # superplastic = float(input("Superplasticizer (kg/m^3): "))
    # coarseagg = float(input("Coarse Aggregate (kg/m^3): "))
    # fineagg = float(input("Fine Aggregate (kg/m^3): "))
    # age = int(input("Age (days): "))

    # Create an array with the input values
    cement = featureDict['cement'] 
    slag = featureDict['slag'] 
    ash = featureDict['ash'] 
    water = featureDict['water'] 
    superplastic = featureDict['superPlastic'] 
    coarseagg = featureDict['coarseAgg'] 
    fineagg = featureDict['fineAgg'] 
    age = featureDict['age'] 
    input_data = np.array([[cement, slag, ash, water, superplastic, coarseagg, fineagg, age]])

    # Make prediction
    prediction = model.predict(input_data)
    return f"{prediction[0]:.2f} MPa"
    

if __name__ == '__main__':
   

    savePath = r'models/RandomForestRegressor.sav'
   
    model2 = load_model(savePath)
    print(predict_concrete_strength(model2))