import datetime
from flask import Flask, render_template, request
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from apscheduler.schedulers.background import BackgroundScheduler


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

url = "https://cropcast.onrender.com/"  
interval = 30  

def reload_website():
    try:
        response = requests.get(url)
        print(f"Reloaded at {datetime.datetime.now().isoformat()}: Status Code {response.status_code}")
    except requests.RequestException as e:
        print(f"Error reloading at {datetime.datetime.now().isoformat()}: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(reload_website, 'interval', seconds=interval)
scheduler.start()

@app.route('/submit', methods=['POST'])
def submit():
    state = request.form.get('option1')
    district = request.form.get('option2')
    total_area = float(request.form.get('hectare'))
    
    # Load and preprocess the dataset
    try:
        df = pd.read_csv("ICRISAT-District Level Data All states.csv")
        max_df = df[df['Dist Name'] == district]
        if max_df.empty:
            return "Error: District data not found in the dataset."
        
        max_df.to_csv("data_ICRI.csv", index=False)
        data = pd.read_csv('data_ICRI.csv')
        data['Year'] = pd.to_datetime(data['Year'], format='%Y')
        data = data.set_index('Year')
        
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        data_numeric = data[numerical_columns].fillna(0)
        if data_numeric.empty:
            return "Error: No numeric data available for the district."
        
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_numeric)
    except Exception as e:
        return f"Data processing error: {e}"

    # Create sequences
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            target = data[i+seq_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    seq_length = 3  
    try:
        X, y = create_sequences(data_scaled, seq_length)
        if len(X) == 0 or len(y) == 0:
            return "Error: Insufficient data for training."
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
    except Exception as e:
        return f"Sequence creation error: {e}"

    # Define and train the model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    input_size = X_train.shape[2]
    hidden_size = 512
    num_layers = 4
    output_size = y_train.shape[1]

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    num_epochs = 200
    batch_size = 512
    try:
        for epoch in range(num_epochs):
            model.train()
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
    except Exception as e:
        return f"Training error: {e}"

    # Model evaluation
    try:
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
        
        test_predictions_unscaled = scaler.inverse_transform(test_predictions.numpy())
        y_test_unscaled = scaler.inverse_transform(y_test.numpy())

        mae = np.mean(np.abs(test_predictions_unscaled - y_test_unscaled))
        print(f"MAE: {mae}")
    except Exception as e:
        return f"Evaluation error: {e}"
    test_predictions_unscaled = scaler.inverse_transform(test_predictions.numpy())
    y_test_unscaled = scaler.inverse_transform(y_test.numpy())

    mae = np.mean(np.abs(test_predictions_unscaled - y_test_unscaled))
    print(mae)
    def predict_future_unscaled(model, scaler, last_sequence, num_predictions):
        model.eval()
        future_predictions = []
        current_sequence = last_sequence.clone()
        
        for _ in range(num_predictions):
            with torch.no_grad():
                prediction = model(current_sequence.unsqueeze(0))
            future_predictions.append(prediction.numpy())
            current_sequence = torch.cat((current_sequence[1:], prediction), dim=0)
        
        future_predictions = np.array(future_predictions).squeeze()
        future_predictions_unscaled = scaler.inverse_transform(future_predictions)
        return future_predictions_unscaled

    last_sequence = X_test[-1]
    num_future_predictions = 3
    future_predictions_unscaled = predict_future_unscaled(model, scaler, last_sequence, num_future_predictions)

        
    raw_data = ""

    column_names = data_numeric.columns
    for i, name in enumerate(column_names):
        raw_data += f"{name}:\n"
        raw_data += f"  Last known value: {y_test_unscaled[-1, i]:.2f}\n"
        raw_data += f"  Predicted next value: {future_predictions_unscaled[0, i]:.2f}\n"
        raw_data += f"  Predicted change: {future_predictions_unscaled[0, i] - y_test_unscaled[-1, i]:.2f}\n"
        raw_data += "\n"

    
    def parse_raw_data(raw_data):
        lines = raw_data.strip().split('\n')
        data = {}
        current_category = None

        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                current_category = line.replace(':', '')
                data[current_category] = {}
            elif ': ' in line:
                key, value = line.split(': ')
                data[current_category][key] = float(value)

        return data

    data = parse_raw_data(raw_data)
    
    crops = {}
    for key in data.keys():
        if key not in ["Dist Code", "State Code"]:
            parts = key.split()
            crop_name = " ".join(parts[:-3]).title()
            metric = parts[-3].lower()
            value = data[key]["Predicted next value"]
            if crop_name not in crops:
                crops[crop_name] = {"area": None, "production": None, "yield": None}
            crops[crop_name][metric] = value

    crops_list = []
    
    crop_prices = {
        "Rice": 3200,  # Updated price
        "Wheat": 2400,  # Updated price
        "Maize": 2200,  # Updated price
        "Kharif Sorghum": 3100,  # Updated price
        "Rabi Sorghum": 3150,  # Updated price
        "Pearl Millet": 2350,  # Updated price
        "Finger Millet": 2600,  # Updated price
        "Barley": 1900,  # Updated price
        "Chickpea": 2600,  # Updated price
        "Pigeonpea": 4500,  # Updated price
        "Minor Pulses": 1850,  # Updated price
        "Groundnut": 3300,  # Updated price
        "Sesamum": 3500,  # Updated price
        "Rapeseed and Mustard": 5700,  # Updated price
        "Safflower": 5900,  # Updated price
        "Castor": 6000,  # Updated price
        "Linseed": 2300,  # Updated price
        "Sunflower": 6400,  # Updated price
        "Soyabean": 4700,  # Updated price
        "Oilseeds": 3500,  # Updated price
        "Cotton": 6700   # Updated price
    }

    for crop, values in crops.items():
        area = values.get("area")
        production = values.get("production")
        crop_yield = round(production / area, 2) if area and production else None
        price = (crop_prices.get(crop, 0)*crop_yield) if crop_yield else crop_prices.get(crop,0)
        crops_list.append({
            "name": crop,
            "area": area,
            "production": production,
            "yield": crop_yield,
            "price": price
        })

    crops_list = sorted(crops_list, key=lambda x: x['production'], reverse=True)

    def greedy_crop_allocation(crops, total_area):
        for crop in crops:
            if crop['area'] > 0:
                crop['yield'] = crop['production'] / crop['area']
                crop['profit_per_area'] = crop['yield'] * crop['price']
            else:
                crop['yield'] = 0
                crop['profit_per_area'] = 0
        
        sorted_crops = sorted(crops, key=lambda x: x['profit_per_area'], reverse=True)
        
        allocation = {}
        remaining_area = total_area
        total_profit = 0
        
        for crop in sorted_crops:
            if remaining_area > 0:
                if crop['profit_per_area'] > 0:  # Only allocate if profit is positive
                    allocated_area = min(crop['area'], remaining_area)
                    if allocated_area >= 0.1:  # Only allocate if area is at least 0.1 hectares
                        allocation[crop['name']] = round(allocated_area, 2)
                        remaining_area -= allocated_area
                        
                        crop_profit = allocated_area * crop['profit_per_area']
                        total_profit += crop_profit
            else:
                break
        
        if remaining_area > 0:
            for crop in sorted_crops:
                if crop['profit_per_area'] > 0:
                    additional_area = min(remaining_area, crop['area'] - allocation.get(crop['name'], 0))
                    if additional_area > 0:
                        if crop['name'] in allocation:
                            allocation[crop['name']] += additional_area
                        else:
                            allocation[crop['name']] = additional_area
                        remaining_area -= additional_area
                        total_profit += additional_area * crop['profit_per_area']
                        if remaining_area == 0:
                            break
        
        for crop in allocation:
            allocation[crop] = round(allocation[crop])
        
        return allocation, total_profit

    result, profit = greedy_crop_allocation(crops_list, total_area)

    allocation_results = [{"crop": crop, "area": area} for crop, area in result.items() if area > 0]

    return render_template('index.html', 
                           state_code=state, 
                           district_code=district, 
                           crops=crops_list, 
                           allocation_results=allocation_results, 
                           total_profit=round(profit, 2),
                           total_area=total_area)

if __name__ == '__main__':
    try:
         app.run()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

