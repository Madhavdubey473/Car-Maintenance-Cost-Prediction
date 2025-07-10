import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

            print("Before loading")

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 brand: str,
                 model: str,
                 engine_type: str,
                 make_year: str,
                 region: str,
                 mileage_range: int,
                 mileage: int,
                 oil_filter: int,
                 engine_oil: int,
                 washer_plug_drain: int,
                 dust_and_pollen_filter: int,
                 whell_alignment_and_balancing: int,
                 air_clean_filter: int,
                 fuel_filter: int,
                 spark_plug: int,
                 brake_fluid: int,
                 brake_and_clutch_oil: int,
                 transmission_fluid: int,
                 brake_pads: int,
                 clutch: int,
                 coolant: int
                 ):
                 
                 
         
         self.brand= brand
         self.model= model
         self.engine_type= engine_type
         self.make_year= make_year
         self.region= region
         self.mileage_range = mileage_range
         self.mileage = mileage
         self.oil_filter = oil_filter
         self.engine_oil = engine_oil
         self.washer_plug_drain = washer_plug_drain
         self.dust_and_pollen_filter = dust_and_pollen_filter
         self.whell_alignment_and_balancing = whell_alignment_and_balancing
         self.air_clean_filter = air_clean_filter
         self.fuel_filter = fuel_filter
         self.spark_plug = spark_plug
         self.brake_fluid = brake_fluid
         self.brake_and_clutch_oil = brake_and_clutch_oil
         self.transmission_fluid = transmission_fluid
         self.brake_pads = brake_pads
         self.clutch = clutch
         self.coolant = coolant


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
         "brand": [self.brand],
         "model":[self.model],
         "engine_type":[self.engine_type],
         "make_year": [self.make_year],
         "region": [self.region],
         "mileage_range": [self.mileage_range ],
         "mileage": [self.mileage ],
         "oil_filter": [self.oil_filter ],
        "engine_oil" : [self.engine_oil ],
         "washer_plug_drain": [self.washer_plug_drain ],
        "dust_and_pollen_filter" : [self.dust_and_pollen_filter ],
         "whell_alignment_and_balancing": [self.whell_alignment_and_balancing ],
         "air_clean_filter": [self.air_clean_filter ],
         "fuel_filter": [self.fuel_filter ],
         "spark_plug": [self.spark_plug ],
         "brake_fluid": [self.brake_fluid ],
         "brake_and_clutch_oil": [self.brake_and_clutch_oil ],
         "transmission_fluid": [self.transmission_fluid ],
         "brake_pads": [self.brake_pads ],
         "clutch": [self.clutch ],
         "coolant": [self.coolant ]
            }

            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e, sys)         
        