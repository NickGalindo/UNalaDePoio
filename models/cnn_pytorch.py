import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Prepare the dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.data = [
            {'label_name': 'casa', 'bbox_x': 486, 'bbox_y': 327, 'bbox_width': 50, 'bbox_height': 45},
            {'label_name': 'casa', 'bbox_x': 497, 'bbox_y': 337, 'bbox_width': 38, 'bbox_height': 28},
            {'label_name': 'casa', 'bbox_x': 519, 'bbox_y': 294, 'bbox_width': 29, 'bbox_height': 21},
            {'label_name': 'casa', 'bbox_x': 570, 'bbox_y': 278, 'bbox_width': 21, 'bbox_height': 17},
            {'label_name': 'casa', 'bbox_x': 574, 'bbox_y': 281, 'bbox_width': 29, 'bbox_height': 19},
            {'label_name': 'casa', 'bbox_x': 518, 'bbox_y': 294, 'bbox_width': 31, 'bbox_height': 20}
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item['label_name']
        bbox_x = item['bbox_x']
        bbox_y = item['bbox_y']
        bbox_width = item['bbox_width']
        bbox_height = item['bbox_height']
        return label, bbox_x, bbox_y, bbox_width, bbox_height

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 2: Design and build the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = NeuralNetwork()

# Step 3: Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 4: Train the neural network
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        labels, bbox_x, bbox_y, bbox_width, bbox_height = batch
        inputs = torch.stack([bbox_x, bbox_y, bbox_width, bbox_height], dim=1)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, torch.ones_like(outputs))
        loss.backward()
        optimizer.step()

# Step 5: Use the neural network for predictions
# Make predictions using the trained model
test_inputs = torch.tensor([[500, 320, 55, 50], [550, 300, 30, 25]])
predictions = model(test_inputs)
print(predictions)

