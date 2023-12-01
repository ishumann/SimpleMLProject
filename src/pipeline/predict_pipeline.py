import sys
# import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object




class PredictPipeline:
    def __init__(self, ):
        pass


    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model= load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction 
        


        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self, 
                 gender, 
                 race_ethnicity: str, 
                 parental_level_of_education: str, 
                 lunch: str, 
                 test_preparation_course: str,
                 writing_score: int, 
                 reading_score: int):
        
        self.gender = gender,
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_frame(self):
        try:
            
            custom_data_as_input_dict = {
                'gender': self.gender[0], 
                'race/ethnicity':[self.race_ethnicity],
                'parental level of education':[self.parental_level_of_education],
                'lunch': [self.lunch], 
                'test preparation course': [self.test_preparation_course],
                'reading score':[self.reading_score],
                'writing score':[self.writing_score]

            }
            return pd.DataFrame(custom_data_as_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
