import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv('./data/spiral.csv')

# Assuming 'X' is the feature matrix and 'y' is the label vector
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Ensure labels are 0 and 1
y = np.where(y == -1, 0, y)  # Convert -1 labels to 0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)



# Define model
class Adaline(nn.Module):
    def __init__(self, input_size):
        super(Adaline, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)

# Initialize model
model = Adaline(X_train.shape[1])

# Define loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for Adaline
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
train_losses = []
test_losses = []
sensitivities = []
specificities = []
accuracies = []

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
        
        # Converting outputs to binary predictions
        y_pred_class = (test_outputs > 0.5).float()
        
        # Calculating accuracy
        accuracy = (y_pred_class == y_test).float().mean()
        accuracies.append(accuracy.item())
        
        # Calculating sensitivity and specificity
        cm = confusion_matrix(y_test.numpy(), y_pred_class.numpy())
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    if (epoch+1) % 10 == 0:
        print(f'Época [{epoch+1}/{num_epochs}], Perda: {loss.item():.4f}, Teste Perda: {test_loss.item():.4f}, Acurácia: {accuracy.item():.4f}')

# Printing final metrics
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
plt.title('Curva de perda de treinamento e teste para Adaline')
plt.legend()
plt.show()
