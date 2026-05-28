import pickle
from flask import Flask, request, render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for a homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None, form_data={})
    else:
        data = CustomData(
            state_electoral_integrity=float(request.form.get('state_electoral_integrity')),
            national_electoral_integrity=float(request.form.get('national_electoral_integrity')),
            political_scale=float(request.form.get('political_scale')),
            election_rigged=int(request.form.get('election_rigged')),
            deadlines_restrictive=int(request.form.get('deadlines_restrictive')),
            voters_waited=int(request.form.get('voters_waited')),
            voters_intimidated=int(request.form.get('voters_intimidated')),
            multiple_ballots_cast=int(request.form.get('multiple_ballots_cast')),
            machines_accurate=int(request.form.get('machines_accurate')),
            records_secure=int(request.form.get('records_secure')),
            votes_counted_quickly=int(request.form.get('votes_counted_quickly')),
            outcome_reflected_popular_will=int(request.form.get('outcome_reflected_popular_will'))
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0], form_data=request.form)

if __name__=="__main__":
    app.run(host="0.0.0.0")

# Go to http://127.0.0.1:5000/predictdata

# Add server side validation
