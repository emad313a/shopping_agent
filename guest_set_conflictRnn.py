import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F


NUM_GUESTS = 24
NUM_SEATS = 24
CONFLICT_COUNT = 40
guest_names = ["Ali", "Alavi", "tome", "stiv", "fati", "javad", "ston", "mahdi", "sobhan", "tohid", 
    "esi", "tiyam", "messi", "ashor", "nahid", "ziba", "mijica", "ana", "ali", "ayoub", 
    "solmaz", "sabori", "mohsa", "sohyla"]

invited_list = np.char.add(
    np.random.choice(guest_names, NUM_GUESTS), 
    [" "] * NUM_GUESTS)
invited_list = np.char.add(invited_list, np.random.choice(guest_names, NUM_GUESTS))
conflict_matrix = np.zeros((NUM_GUESTS, NUM_GUESTS), dtype=int)
conflict_count = 0
while conflict_count < CONFLICT_COUNT:
    i, j = random.sample(range(NUM_GUESTS), 2)
    if conflict_matrix[i, j] == 0:
        conflict_matrix[i, j] = 1
        conflict_count += 1
conflict_matrix = conflict_matrix / np.max(conflict_matrix)

class SeatingModel(nn.Module):
    def __init__(self, num_guests, num_seats):
        super(SeatingModel, self).__init__()
        hidden_layer_size = num_guests * num_guests
        self.lstm = nn.LSTM(input_size=num_guests, hidden_size=hidden_layer_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, num_seats)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.sigmoid(self.fc(x[:, -1, :]))
        return x

def get_conflict(guest1, guest2):
    if guest1 >= 0 and guest1 < NUM_GUESTS and guest2 >= 0 and guest2 < NUM_GUESTS:
        return conflict_matrix[guest1, guest2]
    return 0

def calculate_real_conflict(guest_positions):
    seat_conflict = np.zeros((6, 4))

    for i in range(6):
        for j in range(4):
            current_guest = round(guest_positions[i, j] * (NUM_GUESTS - 1))
            neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
            for ni, nj in neighbors:
                if 0 <= ni < 6 and 0 <= nj < 4:
                    neighbor_guest = round(guest_positions[ni, nj] * (NUM_GUESTS - 1))
                    if neighbor_guest and get_conflict(current_guest, neighbor_guest):
                        seat_conflict[i, j] = 1
                        break
    return seat_conflict

def check_similarity(real_conflict, predicted_conflict):
    return np.sum(real_conflict == predicted_conflict) / real_conflict.size

def conflict_loss(input, output):
    output_reshaped = output.detach().numpy().reshape(6, 4)
    output_normalized = np.where(output_reshaped > 0.5, 1, 0)
    guest_positions = input.detach().numpy().reshape(6, 4)
    real_conflict = calculate_real_conflict(guest_positions)
    real_conflict_flatten = torch.tensor(real_conflict.flatten(), dtype=torch.float32)
    output_flatten = output.flatten()
    return nn.BCELoss()(output_flatten, real_conflict_flatten)

def generate_guest_arrangement():
    return torch.tensor(random.sample([i/NUM_GUESTS for i in range(1, NUM_GUESTS + 1)], NUM_GUESTS), dtype=torch.float32).view(1, 1, -1)

def show_best_result(best_loss, best_arrangement):
    print('Best Loss:', best_loss)
    print('Best Seat Arrangement:', best_arrangement)
    guest_positions = best_arrangement.detach().numpy().reshape(6, 4)
    real_conflict = calculate_real_conflict(guest_positions)
    print('Guest Seat Conflict:\n', real_conflict)

def train_model(model, num_epochs=1000, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    best_arrangement = None
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        input_data = generate_guest_arrangement()
        output = model(input_data)
        loss = conflict_loss(input_data, output)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_arrangement = input_data
    show_best_result(best_loss, best_arrangement)

model = SeatingModel(NUM_GUESTS, NUM_SEATS)
train_model(model, num_epochs=2000)
# Best Lost: 0.16691936552524567
