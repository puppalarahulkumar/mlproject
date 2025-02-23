import os
import sys
import dill 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from src.exception import customException

def save_object(file_path,obj): # this function is called by data transformation
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj) # used to create pickle file.
    except Exception as e:
        pass
def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        
        report={}
        for i in range(len(list(models.values()))):
            
            model=list(models.values())[i]
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_model_score=r2_score(y_train_pred,y_train)
            test_model_score=r2_score(y_test_pred,y_test)
            
            report[list(models.keys())[i]]=test_model_score
            
        return report

    except Exception as e:
        raise customException(e,sys)
        


