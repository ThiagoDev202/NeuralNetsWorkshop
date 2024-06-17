import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, median, stdev, StatisticsError

# Carregar os dados
spiral_data = pd.read_csv('./data/spiral.csv')
X = spiral_data.iloc[:, :2].values
y = spiral_data.iloc[:, 2].values

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Converter para tensores
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Definição da rede MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Função para treinar o modelo
def train_model(model, train_loader, epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

# Função para avaliar o modelo
def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, targets in loader:
            outputs = model(data).squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

# Função para calcular métricas da matriz de confusão
def calc_metrics(outputs, targets):
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    TP = (predicted * targets).sum().item()
    FP = (predicted * (1 - targets)).sum().item()
    TN = ((1 - predicted) * (1 - targets)).sum().item()
    FN = ((1 - predicted) * targets).sum().item()
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    return precision, specificity, sensitivity

# Cálculo de estatísticas
def calculate_stats(metrics):
    stats = {}
    for key in metrics.keys():
        if metrics[key]:
            stats[key] = {
                'mean': mean(metrics[key]),
                'min': min(metrics[key]),
                'max': max(metrics[key]),
                'median': median(metrics[key]),
                'std_dev': stdev(metrics[key])
            }
        else:
            stats[key] = {'mean': 0, 'min': 0, 'max': 0, 'median': 0, 'std_dev': 0}
    return stats

# Treinamento e avaliação
hidden_sizes = [2, 5, 10, 20, 50]
metrics_summary = {'precision': [], 'specificity': [], 'sensitivity': []}
train_accuracies = []
test_accuracies = []

for size in hidden_sizes:
    model = MLP(2, size, 1)
    train_model(model, train_loader, epochs=50)
    train_acc = evaluate_model(model, train_loader)
    test_acc = evaluate_model(model, test_loader)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    for loader in [train_loader, test_loader]:
        model.eval()
        with torch.no_grad():
            for data, targets in loader:
                outputs = model(data).squeeze()
                precision, specificity, sensitivity = calc_metrics(outputs, targets)
                metrics_summary['precision'].append(precision)
                metrics_summary['specificity'].append(specificity)
                metrics_summary['sensitivity'].append(sensitivity)

# Cálculo das estatísticas gerais
final_stats = calculate_stats(metrics_summary)
for key, value in final_stats.items():
    print(f'{key.capitalize()}: Média = {value["mean"]:.2f}, Mínimo = {value["min"]:.2f}, Máximo = {value["max"]:.2f}, '
          f'Mediana = {value["median"]:.2f}, Desvio padrão = {value["std_dev"]:.2f}')

# Gráfico de acurácia
plt.figure(figsize=(10, 5))
plt.plot(hidden_sizes, train_accuracies, label='Treino')
plt.plot(hidden_sizes, test_accuracies, label='Teste')
plt.xlabel('Tamanho da Camada Oculta')
plt.ylabel('Acurácia')
plt.title('Acurácia de Treino vs. Teste')
plt.legend()
plt.grid(True)
plt.show()
