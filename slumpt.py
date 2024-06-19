import numpy as np
import pandas as pd
import pickle
from pyforest import *



def load_model(path):
    model = pickle.load(open(path,'rb'))
    return model

def predict_slump(model, featureDict={}):
    print('Predicting the value...')
 
    cement = featureDict['cement'] 
    slag = featureDict['slag'] 
    ash = featureDict['ash'] 
    water = featureDict['water'] 
    superplastic = featureDict['superPlastic'] 
    silicafumes = featureDict['silicafumes']
    coarseagg = featureDict['coarseAgg'] 
    fineagg = featureDict['fineAgg']  
    input_data = np.array([[cement, slag, ash, water, superplastic, coarseagg, fineagg, silicafumes]])

    # Make prediction
    prediction = model.predict(input_data)
    return f"{prediction[0]:.2f} cm"

if __name__ == '__main__':
    
    savePath = r'models/grad_boost.sav'
 
    model2 = load_model(savePath)
    print(predict_slump(model2))
