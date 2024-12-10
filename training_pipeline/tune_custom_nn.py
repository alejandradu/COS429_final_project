import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pickle

# Load X_resampled and y_resampled data
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/training_pipeline/landmark_datasets/landmark_train_resampled_256_DROP_0.7_0.5_0.5.pkl', 'rb') as f:
    X_resampled, y_resampled = pickle.load(f)

# Convert data to PyTorch tensors
X_resampled = torch.tensor(X_resampled, dtype=torch.float32)
y_resampled = torch.tensor(y_resampled, dtype=torch.long)

# Create a dataset and data loader
dataset = TensorDataset(X_resampled, y_resampled)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define the neural network architecture
class LandmarkNN(nn.Module):
    def __init__(self):
        super(LandmarkNN, self).__init__()
        self.fc1 = nn.Linear(X_resampled.shape[1], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8)
        self.fc8 = nn.Linear(8, 4)
        self.fc9 = nn.Linear(4, 2)
        self.fc10 = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.relu(self.fc8(x))
        x = self.dropout(x)
        x = self.relu(self.fc9(x))
        x = self.fc10(x)
        return x

# Initialize the model, loss function, and optimizer
model = LandmarkNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(X_resampled).squeeze()
        preds = torch.round(torch.sigmoid(outputs))
        accuracy = accuracy_score(y_resampled, preds)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}, Accuracy: {accuracy}")

# Save the model
torch.save(model.state_dict(), '/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/trained_classifiers/landmark_nn_best.pth')