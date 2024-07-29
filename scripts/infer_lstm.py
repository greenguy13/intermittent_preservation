#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import project_utils as pu

torch.manual_seed(42)

# Define dataset
class DecayDataset(Dataset):
  def __init__(self, data, seq_length):
    self.data = data
    self.seq_length = seq_length

  def __len__(self):
    return len(self.data) - self.seq_length + 1

  def __getitem__(self, idx):
    input_seq = self.data[idx:idx + self.seq_length]
    output = self.data[idx + self.seq_length - 1]  # Predict the next time step
    return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)

# Define LSTM model
class DecayLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :])  # Take the output from the last time step
    return out

# Hyperparameters
seq_length = 8
batch_size = 8
hidden_size = 8
num_layers = 3
learning_rate = 0.001
num_epochs = 250

# Train the model
def train_model_lstm(data):

  # # Check if GPU is available
  # if torch.cuda.is_available():
  #   device = torch.device("cuda")
  #   debug("GPU is available and being used.")
  # else:
  #   device = torch.device("cpu")
  #   debug("GPU is not available, using CPU instead.")

  #Scale data
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data.values)

  #Generate data loader
  dataset = DecayDataset(scaled_data, seq_length)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  #Initiate LSTM model
  input_size = data.shape[1]
  output_size = data.shape[1]
  model = DecayLSTM(input_size, hidden_size, num_layers, output_size)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  #Train model
  for epoch in range(num_epochs):
    model.train()
    for inputs, targets in dataloader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

  return model

def forecast_decay_lstm(model, data, steps):
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data.values)

  with torch.no_grad():
    # Take the last 'seq_length' data points as input for prediction
    test_input = torch.tensor(scaled_data[-seq_length:], dtype=torch.float32).unsqueeze(0)
    predictions = []
    for _ in range(steps):  # Forecast for time steps
      predicted_scaled = model(test_input)
      predictions.append(predicted_scaled.numpy())
      # Update the input for the next prediction
      test_input = torch.cat((test_input[:, 1:, :], predicted_scaled.unsqueeze(1)), dim=1)

  # Inverse transform to get the actual decay rates
  predicted_data = scaler.inverse_transform(np.concatenate(predictions, axis=0))

  # Convert to data frame
  predicted_data = pd.DataFrame(predicted_data, columns=data.columns)

  return predicted_data

def debug(msg, robot_id=0):
  pu.log_msg('robot', robot_id, msg, debug=True)