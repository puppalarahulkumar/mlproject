import os
import sys
import dill 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from src.exception import customException
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path,obj): # this function is called by data transformation
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj) # used to create pickle file.
    except Exception as e:
        raise customException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        
        report={}
        for i in range(len(list(models.values()))):
            
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]

            rs=RandomizedSearchCV(model,param,cv=3)
            rs.fit(x_train,y_train)

            model.set_params(**rs.best_params_)
            model.fit(x_train,y_train)

            # model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_model_score=r2_score(y_train_pred,y_train)
            test_model_score=r2_score(y_test_pred,y_test)
            
            report[list(models.keys())[i]]=test_model_score
            
        return report

    except Exception as e:
        raise customException(e,sys)
        
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise customException(e,sys)


