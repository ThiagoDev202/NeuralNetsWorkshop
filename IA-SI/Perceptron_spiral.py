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
class SimplePerceptron(nn.Module):
    def __init__(self, input_size):
        super(SimplePerceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)  # Using sigmoid activation for better gradient flow
        return x

# Initialize model
model = SimplePerceptron(X_train.shape[1])

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
train_losses = []
test_losses = []
accuracies = []
specificities = []
sensitivities = []

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
        # Avaliação no conjunto de teste
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
        
        # Convertendo probabilidades em saídas binárias
        y_pred_class = (test_outputs > 0.5).float()
        
        # Calculando a matriz de confusão
        tn, fp, fn, tp = confusion_matrix(y_test.numpy(), y_pred_class.numpy()).ravel()
        
        # Calculando métricas
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        # Armazenando métricas
        accuracies.append(accuracy)
        specificities.append(specificity)
        sensitivities.append(sensitivity)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}')


# Plotar a curva de perda de treinamento e teste
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve for Simple Perceptron')
plt.legend()
plt.show()

# Estatísticas descritivas
print('Métricas após 100 épocas:')
print(f'Accuracy - Mean: {np.mean(accuracies):.4f}, Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}, Median: {np.median(accuracies):.4f}, Std: {np.std(accuracies):.4f}')
print(f'Specificity - Mean: {np.mean(specificities):.4f}, Min: {np.min(specificities):.4f}, Max: {np.max(specificities):.4f}, Median: {np.median(specificities):.4f}, Std: {np.std(specificities):.4f}')
print(f'Sensitivity - Mean: {np.mean(sensitivities):.4f}, Min: {np.min(sensitivities):.4f}, Max: {np.max(sensitivities):.4f}, Median: {np.median(sensitivities):.4f}, Std: {np.std(sensitivities):.4f}')
