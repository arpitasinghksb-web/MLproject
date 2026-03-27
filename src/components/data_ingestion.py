
# import os
# import sys
# import pandas as pd

# from dataclasses import dataclass
# from sklearn.model_selection import train_test_split

# from src.exception import CustomException
# from src.logger import logging
# from src.components.data_transformation import DataTransformation

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer


# @dataclass
# class DataIngestionConfig:
#     train_data_path: str = os.path.join('artifacts', 'train.csv')
#     test_data_path: str = os.path.join('artifacts', 'test.csv')
#     raw_data_path: str = os.path.join('artifacts', 'data.csv')


# class DataIngestion:
#     def __init__(self):
#         self.ingestion_config = DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         logging.info("Entered the data ingestion component")
#         try:
#             current_dir = os.getcwd()
#             file_path = os.path.join(current_dir, 'notebook', 'data', 'stud.csv')

#             df = pd.read_csv(file_path)
#             logging.info('Dataset loaded successfully')

#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

#             df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

#             logging.info("Train-test split started")

#             train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

#             train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
#             test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

#             logging.info("Data ingestion completed successfully")

#             return (
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#             )

#         except Exception as e:
#             raise CustomException(e, sys)


# if __name__ == "__main__":
#     try:
#         obj = DataIngestion()
#         train_data, test_data = obj.initiate_data_ingestion()

#         data_transformation = DataTransformation()
#         train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

#         modeltrainer = ModelTrainer()
#         print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

#     except Exception as e:
#         raise CustomException(e, sys)


import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    source_data_path: str = os.path.join('notebook', 'data', 'stud.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered Data Ingestion")

            if not os.path.exists(self.ingestion_config.source_data_path):
                raise CustomException("Source file not found", sys)

            df = pd.read_csv(self.ingestion_config.source_data_path)

            os.makedirs('artifacts', exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        print("Full Pipeline Started")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print("Data Ingestion Done")

        from src.components.data_transformation import DataTransformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
        print("Data Transformation Done")

        from src.components.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        r2_score = trainer.initiate_model_trainer(train_arr, test_arr)
        print("Model Training Done")

        print("Final R2 Score:", r2_score)

    except Exception as e:
        print("Error:", e)