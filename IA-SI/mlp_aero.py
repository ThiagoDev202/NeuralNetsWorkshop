import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv('data/aerogerador.dat', sep='\s+', header=None)

# Assuming 'X' is the feature matrix and 'y' is the label vector
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Ensure labels are 0 and 1
y = np.where(y > 0, 1, 0)  # Convert to binary if not already

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define model
class AerogeradorModel(nn.Module):
    def __init__(self, input_size):
        super(AerogeradorModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize model
model = AerogeradorModel(X_train.shape[1])

# Define loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Metric lists
accuracies = []
sensitivities = []
specificities = []

# Train the model
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
        
        # Calculate binary predictions and metrics
        predicted = (test_outputs > 0.5).float()
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        accuracies.append(accuracy)
        
        # Confusion matrix for calculating sensitivity and specificity
        cm = confusion_matrix(y_test.numpy(), predicted.numpy(), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-7)  # Avoid division by zero
        sensitivities.append(sensitivity)
        specificity = tn / (tn + fp + 1e-7)  # Avoid division by zero
        specificities.append(specificity)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Print final metric summaries
print(f'Média Acurácia: {np.mean(accuracies):.4f}')
print(f'Mínimo Acurácia: {np.min(accuracies):.4f}')
print(f'Máximo Acurácia: {np.max(accuracies):.4f}')
print(f'Mediana Acurácia: {np.median(accuracies):.4f}')
print(f'Desvio padrão Acurácia: {np.std(accuracies):.4f}')

print(f'Média Sensibilidade: {np.mean(sensitivities):.4f}')
print(f'Média Especificidade: {np.mean(specificities):.4f}')

# Plotting training and testing loss curves
plt.plot(train_losses, label='Treinamento Perda')
plt.plot(test_losses, label='Teste Perda')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Curva de perda de treinamento e teste para Modelo Aerogerador')
plt.legend()
plt.show()
