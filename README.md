# Neural Machine Learning Algorithms

This repository contains multiple Python scripts to perform binary classification using various neural network models implemented with PyTorch. The models include a Multi-Layer Perceptron (MLP), an Adaline model, and a Simple Perceptron. The code demonstrates data preprocessing, model training, evaluation, and performance metric calculation.

Overview

    Data: The dataset used is the spiral.csv file, containing two features and a binary target label.
    Libraries: The implementation uses PyTorch for model building, training, and evaluation. Other libraries include pandas, numpy, and matplotlib for data manipulation and visualization.
    Models: The models implemented in this repository are:
        Multi-Layer Perceptron (MLP)
        Adaline
        Simple Perceptron

Files

    MLP Model (mlp_model.py):
        Implements a Multi-Layer Perceptron with customizable hidden layer sizes.
        Trains the model using the binary cross-entropy loss function and the Adam optimizer.
        Evaluates the model using accuracy, precision, specificity, and sensitivity.
        Provides a summary of statistics for these metrics.

    Adaline Model (adaline_model.py):
        Implements an Adaline (Adaptive Linear Neuron) model using Mean Squared Error (MSE) as the loss function.
        Trains the model using Stochastic Gradient Descent (SGD) optimizer.
        Computes evaluation metrics including accuracy, sensitivity, and specificity.
        Outputs descriptive statistics for accuracy and loss values.

    Simple Perceptron (simple_perceptron.py):
        Implements a basic Perceptron model with sigmoid activation for binary classification.
        Uses Binary Cross Entropy (BCE) as the loss function and the SGD optimizer.
        Tracks training and testing loss over epochs.
        Evaluates the model's accuracy, sensitivity, and specificity after each epoch.

Installation

    Clone the repository:

bash

git clone https://github.com/your-repo/spiral-classification.git
cd spiral-classification

    Install the required dependencies:

bash

pip install -r requirements.txt

    Run the scripts to train and evaluate the models:

bash

python mlp_model.py
python adaline_model.py
python simple_perceptron.py

Usage
Data Preprocessing

    The dataset is normalized using StandardScaler.
    The data is split into training and test sets using an 80/20 ratio.
    PyTorch tensors are created for use with the neural network models.

Model Training

    The models are trained for a predefined number of epochs (default is 100).
    Each model uses a different optimizer and loss function suited to its architecture.

Evaluation

    The models are evaluated on both training and test datasets.
    Evaluation metrics include accuracy, sensitivity, specificity, precision, and other statistical summaries.
    The training and test loss curves are plotted for visual inspection.

Results

    The scripts output evaluation metrics and statistics for each model.
    Final accuracy, sensitivity, and specificity values are reported after training is complete.
    Training and test loss curves are plotted to visualize model performance over time.

Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or bug fixes.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

    This project uses the PyTorch framework, which is open-source software developed by the PyTorch team.
