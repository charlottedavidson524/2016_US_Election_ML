import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = param[list(models.keys())[i]]

            logging.info(f"Starting GridSearchCV for {model_name}")
            gs = GridSearchCV(model, para, cv=5, scoring='r2')
            gs.fit(X_train, y_train)
            logging.info(f"{model_name} best params: {gs.best_params_}")
            logging.info(f"{model_name} best CV R²: {gs.best_score_:.4f}")
            #best_model = gs.best_estimator_

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            logging.info(f"{model_name} test R²: {test_model_score:.4f}")

            report[list(models.keys())[i]] = test_model_score

        return report 
    
    except Exception as e:
        raise CustomException(e, sys)
        