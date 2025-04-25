from django.shortcuts import render
import pickle
import numpy as np
import os
import json
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
    future_data = None
    chart_data = None  # Renamed from context to avoid confusion
    
    if request.method == 'POST':
        print("Form submitted!")  # Debug print
        print("POST data:", request.POST)  # Debug print

        state = request.POST.get('state')
        print("State:", state)  # Debug print
        
        try:
            future_data = predict_future_crimes_with_types(state, data, model, scaler, le)
            
            if isinstance(future_data, str):  # If it returned an error message
                prediction = {"error": future_data}
            else:
                # Extract the predictions for the specified year
                prediction_data = future_data[['Year', 'Rape', 'K&A', 'DD', 'AoW', 'DV', 'Predicted Total Crimes']]
                # Convert prediction data to dictionary for rendering
                prediction = prediction_data.to_dict(orient='records')

                chart_data = {
                    'labels': [item['Year'] for item in prediction],
                    'rape': [item['Rape'] for item in prediction],
                    'ka': [item['K&A'] for item in prediction],
                    'dd': [item['DD'] for item in prediction],
                    'aow': [item['AoW'] for item in prediction],
                    'dv': [item['DV'] for item in prediction],
                    'predicted': [item['Predicted Total Crimes'] for item in prediction],

                }
                

        except Exception as e:
            prediction = {"error": str(e)}
    
    # Combine all context variables
    context = {
        'prediction': prediction,
        'labels': chart_data['labels'] if chart_data else None,
        'rape': chart_data['rape'] if chart_data else None,
        'ka': chart_data['ka'] if chart_data else None,
        'dd': chart_data['dd'] if chart_data else None,
        'aow': chart_data['aow'] if chart_data else None,
        'dv': chart_data['dv'] if chart_data else None,
        'predicted': chart_data['predicted'] if chart_data else None,
        'selected_state': state if request.method == 'POST' else None,  # Add this line
    }
    
    return render(request, 'home.html', context)

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

# <script id="labels-data" type="application/json">{{ labels|json_script:"labels-data" }}</script>
# <script id="labels-data" type="application/json">{{ labels|json_script:"labels-data" }}</script>
