# Electoral Integrity Prediction

A machine learning web app that predicts perceptions of electoral integrity index score based on survey responses about electoral processes. Built with a scikit-learn/XGBoost regression pipeline and served as a Flask web app.

## Overview

A regression model was trained on survey data capturing voter perceptions of electoral integrity. This covered areas like voter intimidation, ballot security, machine accuracy and outcome legitimaxy. The best performing model is selected automatically via cross-validated hyperparametertuning and served through a web interface.

## Tech Stack

- ML: scikit-learn, XGBoost, pandas, numpy
- Web: Flask, Gunicorn
- Containerisation: Docker
- Deployment: AWS

## ML Pipeline

1. Data ingesion: reads survey CSV and splits into train/test sets
2. Data transformation: Likert-scale responses encoded to ordinal integers, then median imputation and standard scaling applied
3. Model training: Six models evaluated using GridSearchCV (Linear Regression, Ridge, Lasso, ElasticNet, Random Forest and XGBoost). Best one saved to `artifacts/model.pkl`

## Access

#### Live Application

Coming soon :D

#### Run Locally (Python)

```bash
pip install -r requirements.txt
python src/components/data_ingestion.py  # Trains model
python app.py
```

Then visit `http://127.0.0.1:5000/predictdata`

#### Run Locally (Docker)

`docker build -t fraud-detection-ml:latest .`
`docker run -p 5000:5000 fraud-detection-ml:latest`

Then visit `http://localhost:5000`
