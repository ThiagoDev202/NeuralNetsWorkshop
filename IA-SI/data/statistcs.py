import matplotlib.pyplot as plt

# Dados dos classificadores
data = {
    "Perceptron": {
        "Accuracy": [0.8015, 0.5000, 1.0000, 0.8000, 0.1969],
        "Sensitivity": [0.8675],
        "Specificity": [0.7575]
    },
    "MLP": {
        "Accuracy": [0.9770, 0.6667, 1.0000, 1.0000, 0.0478],
        "Sensitivity": [0.0500],
        "Specificity": [1.0000]
    },
    "Adaline": {
        "Accuracy": [0.6290, 0.0500, 1.0000, 0.6000, 0.2731],
        "Sensitivity": [0.8487],
        "Specificity": [0.4825]
    }
}

# Configurando a figura
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
fig.suptitle('Comparative Box-Plot of Classifiers on Iris Dataset')

# Plotando
for idx, (key, values) in enumerate(data.items()):
    axes[idx].boxplot([values["Accuracy"], values["Sensitivity"], values["Specificity"]],
                      labels=['Accuracy', 'Sensitivity', 'Specificity'])
    axes[idx].set_title(f'{key}')
    axes[idx].set_ylim(0, 1.05)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()