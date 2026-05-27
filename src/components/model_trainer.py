import os 
import sys
from dataclasses import dataclass 

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1], 
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(max_iter=10000),
                "ElasticNet": ElasticNet(max_iter=10000),
                "Random Forest": RandomForestRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(random_state=42),
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Get best model score from the dictionary 
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from the dictionary 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # Quality check
            if best_model_score < 0.6:
                raise CustomException("No decent quality model", sys)
            
            logging.info(f"Best model: {best_model_name} with R² {best_model_score:.4f}")

            # Save the best model as a pkl file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)