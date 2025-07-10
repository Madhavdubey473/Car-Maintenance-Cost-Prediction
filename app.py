from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            brand = request.form.get('brand'),
            model = request.form.get('model'),
            engine_type = request.form.get('engine_type'),
            make_year = request.form.get('make_year'),
            region = request.form.get('region'),
            mileage_range = request.form.get('mileage_range'),
            mileage = request.form.get('mileage'),
            oil_filter = request.form.get('oil_filter'),
            engine_oil = request.form.get('engine_oil'),
            washer_plug_drain = request.form.get('washer_plug_drain'),
            dust_and_pollen_filter = request.form.get('dust_and_pollen_filter'),
            whell_alignment_and_balancing = request.form.get('whell_alignment_and_balancing'),
            air_clean_filter = request.form.get('air_clean_filter'),
            fuel_filter = request.form.get('fuel_filter'),
            spark_plug = request.form.get('spark_plug'),
            brake_fluid = request.form.get('brake_fluid'),
            brake_and_clutch_oil = request.form.get('brake_and_clutch_oil'),
            transmission_fluid = request.form.get('transmission_fluid'),
            brake_pads = request.form.get('brake_pads'),
            clutch = request.form.get('clutch'),
            coolant = request.form.get('coolant')




        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")    