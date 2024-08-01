from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    state = request.form.get('option1')
    district = request.form.get('option2')
    total_area = float(request.form.get('hectare'))
    
    df = pd.read_csv("ICRISAT-District Level Data All states.csv")

    max_df = df[df['Dist Name'] == district]

    print(max_df)

    max_df.to_csv("data_ICRI.csv", index=False)
    print("Result saved to 'data_ICRI.csv'")
    
    

    data = pd.read_csv('data_ICRI.csv')  
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    data = data.set_index('Year')

    numerical_columns = data.select_dtypes(include=[np.number]).columns
    data_numeric = data[numerical_columns]
    data_numeric = data_numeric.fillna(0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_numeric)

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
    X, y = create_sequences(data_scaled, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    print(X.shape)
    print(y.shape)


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
    hidden_size = 64
    num_layers = 2
    output_size = y_train.shape[1]

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())


    num_epochs = 200
    batch_size = 64

    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)

    test_predictions_unscaled = scaler.inverse_transform(test_predictions.numpy())
    y_test_unscaled = scaler.inverse_transform(y_test.numpy())

    mae = np.mean(np.abs(test_predictions_unscaled - y_test_unscaled))
    print(f'Mean Absolute Error (unscaled): {mae:.4f}')

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
    num_future_predictions = 5
    future_predictions_unscaled = predict_future_unscaled(model, scaler, last_sequence, num_future_predictions)

    print("Future predictions (unscaled):")
    for i, pred in enumerate(future_predictions_unscaled):
        print(f"Year {i+1}: {pred}")
        
    raw_data = ""

    column_names = data_numeric.columns
    for i, name in enumerate(column_names):
        raw_data += f"{name}:\n"
        raw_data += f"  Last known value: {y_test_unscaled[-1, i]:.2f}\n"
        raw_data += f"  Predicted next value: {future_predictions_unscaled[0, i]:.2f}\n"
        raw_data += f"  Predicted change: {future_predictions_unscaled[0, i] - y_test_unscaled[-1, i]:.2f}\n"
        raw_data += "\n"

    print(raw_data)

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
        "Rice": 20.5,
        "Wheat": 18.75,
        "Maize": 15.0,
        "Kharif Sorghum": 12.0,
        "Rabi Sorghum": 13.0,
        "Pearl Millet": 10.5,
        "Finger Millet": 14.0,
        "Barley": 16.0,
        "Chickpea": 50.0,
        "Pigeonpea": 60.0,
        "Minor Pulses": 45.0,
        "Groundnut": 70.0,
        "Sesamum": 80.0,
        "Rapeseed and Mustard": 65.0,
        "Safflower": 55.0,
        "Castor": 40.0,
        "Linseed": 52.0,
        "Sunflower": 75.0,
        "Soyabean": 35.0,
        "Oilseeds": 60.0,
        "Sugarcane": 3.0,  
        "Cotton": 80.0,
        "Fruits": 40.0,  
        "Vegetables": 20.0,  
        "Potatoes": 10.0,
        "Onion": 12.0,
        "Fodder": 5.0  
    }

    for crop, values in crops.items():
        area = values.get("area")
        production = values.get("production")
        crop_yield = round(production / area, 2) if area and production else None
        price = crop_prices.get(crop, 0)
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
                allocated_area = min(crop['area'], remaining_area)
                allocation[crop['name']] = round(allocated_area, 2)
                remaining_area -= allocated_area
                
                crop_profit = allocated_area * crop['profit_per_area']
                total_profit += crop_profit
            else:
                break
        
        return allocation, total_profit

    result, profit = greedy_crop_allocation(crops_list, total_area)

    allocation_results = [{"crop": crop, "area": area} for crop, area in result.items()]

    return render_template('index.html', 
                           state_code=state, 
                           district_code=district, 
                           crops=crops_list, 
                           allocation_results=allocation_results, 
                           total_profit=round(profit, 2),
                           total_area=total_area)

if __name__ == '__main__':
    app.run(debug=True)
