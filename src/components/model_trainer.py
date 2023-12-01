import os
import sys



from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer(object):
    """docstring for ModelTrainer."""
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trianer(self, train_array, test_array):
        try:
            logging.info("splitting training and testing data")
            X_train, X_test, y_train, y_test = (
            train_array[:,:-1],
            test_array[:,:-1],
            train_array[:,-1],
            test_array[:,-1]
            )

            models = {
                        "Linear Regression": LinearRegression(),
                        "Lasso": Lasso(),
                        "Ridge": Ridge(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(),
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'n_estimators': [8,16,32,64,128,256]
                    # 'splxxitter': ['best', 'random'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    # 'max_depth' : [4,5,6,7,8]
                    },
                "Random Forest Regressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    # 'max_depth' : [4,5,6,7,8]
                    },
                "Gradient Boosting": {
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],

                },

                'Linear Regression':{},

                "Ridge":{
                    'alpha': [0.01, .03, .1,.3,1,10,30, 100],

                },

                "Lasso":{
                    'alpha': [0.01, .03, .1,.3,1,10,30, 100],
                    

                },



                'XGBRegressor': {
                'n_estimators': [8,16,32,64,128,256],
                'learning_rate': [0.01,0.05,0.1,0.3],

                },

                'CatBoosting Regressor': {
                    'iterations': [8,16,32,64,128,256],
                    'learning_rate': [0.01,0.05,0.1,0.3],
                    'depth': [6,8,10]
                },

                'AdaBoost Regressor': {
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [.001,0.01,0.05,0.1,0.3],
                    # 'loss': ['linear', 'square', 'exponential']
                },





            }


 
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test= X_test, y_test = y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            
            best_model = models[best_model_name]

            if best_model_score< 0.6:
                raise CustomException("no best model found, All models are below 0.6.", sys)
            
            logging.info(f"Model that is the Best on the both train and test dataset is:  {best_model_name}") 


            save_object(file_path = self.model_trainer_config.trained_model_file_path, obj = best_model)

            predicted = best_model.predict(X_test)            
            r2_square = r2_score(y_test, predicted)
            return r2_square 



        except Exception as e:
            raise CustomException(e,sys)

    







