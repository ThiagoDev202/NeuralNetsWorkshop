import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar a base de dados Iris
iris_data_path = 'C:\\Users\\Windows\\Downloads\\IA-SI\\iris\\iris.data'
iris_data = pd.read_csv(iris_data_path, header=None)

# Convertendo os rótulos para categorias numéricas
iris_data[4] = iris_data[4].astype('category').cat.codes

# Preparar matrizes de características e vetores de rótulo
X = iris_data.iloc[:, :-1].values
y = iris_data[4].values

# Normalização das características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertendo para tensores PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define o modelo MLP
class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)  # Primeira camada escondida
        self.layer2 = nn.Linear(50, 20)          # Segunda camada escondida
        self.out = nn.Linear(20, num_classes)    # Camada de saída

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.out(x)
        return x

# Initialize model
model = MLPModel(X_train.shape[1], 3)  # 3 classes para Iris

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Metric lists
accuracies = []
sensitivities = []
specificities = []

# Train the model
num_epochs = 100
train_losses = []
test_losses = []
times = []

total_start_time = time.time()

for epoch in range(num_epochs):
    start_time = time.time()
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
        _, predicted = torch.max(test_outputs, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        accuracies.append(accuracy)
        
        # Confusion matrix for calculating sensitivity and specificity
        cm = confusion_matrix(y_test.numpy(), predicted.numpy())
        tn, fp, fn, tp = cm.ravel(order='C')[:4]
        sensitivity = tp / (tp + fn + 1e-7)
        sensitivities.append(sensitivity)
        specificity = tn / (tn + fp + 1e-7)
        specificities.append(specificity)

    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Tempo de Época: {elapsed_time:.4f} s')

total_time = time.time() - total_start_time
print(f'Tempo Total de Treinamento: {total_time:.4f} s')

# Plotting training and testing loss curves
plt.plot(train_losses, label='Treinamento Perda')
plt.plot(test_losses, label='Teste Perda')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Curva de perda de treinamento e teste para Modelo MLP Iris')
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
