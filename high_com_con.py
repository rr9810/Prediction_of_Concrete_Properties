import numpy as np
import pandas as pd
import pickle
from pyforest import *



def load_model(path):
    model = pickle.load(open(path,'rb'))
    return model

def predict_highconcretestrength(model, featureDict={}):
    print('Predicting the value...')
    # 
    # Create an array with the input values
    cement = featureDict['cement'] 
    slag = featureDict['slag'] 
    ash = featureDict['ash'] 
    water = featureDict['water'] 
    superplastic = featureDict['superPlastic']
    silicafumes = featureDict['silicafumes'] 
    coarseagg = featureDict['coarseAgg'] 
    fineagg = featureDict['fineAgg'] 
    age = featureDict['age'] 
    input_data = np.array([[cement, slag, ash, water, superplastic, silicafumes, coarseagg, fineagg, age]])

    # Make prediction
    prediction = model.predict(input_data)
    return f"{prediction[0]:.2f} MPa"

if __name__ == '__main__':
   
    savePath = r'models/linear.sav'
    
    model2 = load_model(savePath)
    print(predict_highconcretestrength(model2))
