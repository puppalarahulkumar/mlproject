# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
from sklearn.metrics import mean_squared_error, r2_score

from src.utils import save_object,evaluate_models

from src.logger import logging
import os
import sys
from dataclasses import dataclass
from src.utils import save_object
from src.exception import customException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test.")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            model_report:dict=evaluate_models(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models)

            # to get the best model score
            best_model_score=max(model_report.values())

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise customException("No Best Model Found.")
            logging.info("the best model is found on both training and testing dataset.")

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )            
            predicted=best_model.predict(X_test)

            r2_square= r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise customException(e,sys)

