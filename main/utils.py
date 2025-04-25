import pandas as pd

def predict_future_crimes_with_types(state_name, data, model, scaler, label_encoder, start_year=2012, end_year=2030):
    crime_cols = ['Rape', 'K&A', 'DD', 'AoW', 'DV']

    try:
        feature_cols = scaler.feature_names_in_.tolist()
    except AttributeError:
        raise ValueError("Scaler is missing 'feature_names_in_' — was it fitted on a DataFrame with column names?")

    state_encoded = label_encoder.transform([state_name])[0]
    state_data = data[data['State_encoded'] == state_encoded].copy()

    trends = {
        col: state_data[col].diff().rolling(5).mean().iloc[-1]
        for col in crime_cols
    }

    last_row = state_data.iloc[-1]
    future_data = []

    for year in range(start_year, end_year + 1):
        new_row = {
            'Unnamed: 0': last_row['Unnamed: 0'] + (year - last_row['Year']),
            'Year': year,
            'State_encoded': state_encoded
        }

        for col in crime_cols:
            last_value = future_data[-1][col] if future_data else last_row[col]
            new_row[col] = max(0, last_value + trends[col])

        future_data.append(new_row)

    future_df = pd.DataFrame(future_data)

    try:
        X_future = future_df[feature_cols]
    except KeyError as e:
        print("Mismatch in features. Available in future_df:", future_df.columns.tolist())
        print("Expected by scaler:", feature_cols)
        raise e

    X_scaled = scaler.transform(X_future)
    predictions = model.predict(X_scaled)
    future_df['Predicted Total Crimes'] = predictions

    return future_df[['Year'] + crime_cols + ['Predicted Total Crimes']]
