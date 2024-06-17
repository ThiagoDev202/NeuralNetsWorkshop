import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregando o conjunto de dados
data = pd.read_csv('data/aerogerador.dat', sep=r'\s+', header=None)

# Convertendo os rótulos para binário (0 ou 1)
data.iloc[:, -1] = np.where(data.iloc[:, -1] > 0, 1, 0)

# Dividindo em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values, test_size=0.2, random_state=42)

# Convertendo para tensores PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Definindo o modelo
class SimplePerceptron(nn.Module):
    def __init__(self, input_size):
        super(SimplePerceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)

# Inicializando o modelo
model = SimplePerceptron(X_train.shape[1])

# Definindo a função de perda e o otimizador
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loop de treinamento
num_epochs = 100
train_losses = []
test_losses = []
# Metric lists
accuracies = []
sensitivities = []
specificities = []


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
        print(f'Época [{epoch+1}/{num_epochs}], Perda de Treinamento: {loss.item():.4f}, Perda de Teste: {test_loss.item():.4f}')

# Plotando as curvas de perda
plt.plot(train_losses, label='Treinamento')
plt.plot(test_losses, label='Teste')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Curva de Perda para Modelo Aerogerador')
plt.legend()
plt.show()


# Print final metric summaries
print(f'Média Acurácia: {np.mean(accuracies):.4f}')
print(f'Mínimo Acurácia: {np.min(accuracies):.4f}')
print(f'Máximo Acurácia: {np.max(accuracies):.4f}')
print(f'Mediana Acurácia: {np.median(accuracies):.4f}')
print(f'Desvio padrão Acurácia: {np.std(accuracies):.4f}')

print(f'Média Sensibilidade: {np.mean(sensitivities):.4f}')
print(f'Média Especificidade: {np.mean(specificities):.4f}')