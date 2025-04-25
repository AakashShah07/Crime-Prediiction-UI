from django.shortcuts import render
import pickle
import numpy as np
import os
from django.conf import settings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .utils import predict_future_crimes_with_types
# Load model files from the project's base directory
BASE_DIR = settings.BASE_DIR
model_path = os.path.join(BASE_DIR, 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
label_encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
csv_data_path = os.path.join(BASE_DIR,  'data.csv') 


with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(label_encoder_path, 'rb') as f:
    le = pickle.load(f)


data = pd.read_csv(csv_data_path)
if 'State_encoded' not in data.columns:
    data['State_encoded'] = le.transform(data['State'])


def home(request):
    prediction = None
    future_data= None
    if request.method == 'POST':
        print("Form submitted!")  # Debug print
        print("POST data:", request.POST)  # Debug print

        state = request.POST.get('state')
        # year = int(request.POST.get('year'))  # Convert to integer
        print("State:", state)  # Debug print
        try:
            future_data = predict_future_crimes_with_types(state, data, model, scaler, le)
            if isinstance(future_data, str):  # If it returned an error message
                    prediction = future_data
            else:
                    # Extract the predictions for the specified year
                    prediction_data = future_data[['Year', 'Rape', 'K&A', 'DD', 'AoW', 'DV', 'Predicted Total Crimes']]
                    # Convert prediction data to dictionary for rendering
                    prediction = prediction_data.to_dict(orient='records')
                    print(prediction)

        except Exception as e:
            prediction = {"error": str(e)}

    return render(request, 'home.html', {'prediction': prediction})


def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')
