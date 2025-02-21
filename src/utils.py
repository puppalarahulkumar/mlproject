import os
import sys
import dill 
import numpy as np
import pandas as pd

from src.exception import customException

def save_object(file_path,obj): # this function is called by data transformation
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj) # used to create pickle file.
    
    except Exception as e:
        pass
