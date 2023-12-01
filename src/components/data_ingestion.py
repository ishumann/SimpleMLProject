import os 
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTranformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("read the dataset as a dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header = True)
            logging.info("train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True)

            logging.info("Ingestion of the data has completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

        except Exception as e:
            logging.info(" error happend in data ingestion method")
            raise CustomException(e,sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    DataTranformation = DataTranformation()
    train_arr, test_arr, pkl_file = DataTranformation.initiate_data_tranformation(train_data, test_data)
    logging.info(f"Data Ingestion and transformation is completed successfully and the data is stored in {train_data},  {test_data} and {pkl_file} ")



    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trianer(train_arr, test_arr))
