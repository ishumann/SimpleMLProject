import sys
import os
from dataclasses import dataclass


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.

    This class holds the configuration parameters for data transformation operations.
    """

    # Rest of the class implementation goes here

    preprocessor_ob_file_path = os.path.join("artifacts","preprocessor.pkl")



class DataTranformation:


    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformers_object(self):
        try:
            numerical_columns = ["writing score", "reading score"]

            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"numerical Columns: {numerical_columns}")
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )            

            logging.info(f"Categorical Columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipelines", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data load of train and test is  successfully")

            logging.info("Obtaining Preprocessor object")

            preprocessing_obj = self.get_data_transformers_object()

            target_column_name =  "math score"
            numaerical_columns = ["writing score", "reading score"]

            input_fearture_train_df = train_df.drop(target_column_name, axis=1)
            target_fearture_train_df = train_df[target_column_name]
            
            input_fearture_test_df = test_df.drop(target_column_name, axis=1)
            target_fearture_test_df = test_df[target_column_name]

            logging.info("Fitting the preprocessor object on train and testing data") 


            input_fearture_train_arr = preprocessing_obj.fit_transform(input_fearture_train_df)
            input_fearture_test_arr = preprocessing_obj.fit_transform(input_fearture_test_df)


            train_arr = np.c_[input_fearture_train_arr,np.array( target_fearture_train_df)]

            test_arr = np.c_[input_fearture_test_arr,np.array( target_fearture_test_df)]

            logging.info("Saved the preprocessor object")

            save_object(
                
                file_path = self.data_transformation_config.preprocessor_ob_file_path, obj = preprocessing_obj
                        
            )
                        


            return ( train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path )

        except Exception as e:
             raise CustomException(e,sys)
