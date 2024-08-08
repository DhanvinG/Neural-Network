Neural Network for Iris Flower Classification

Overview

This project implements a simple neural network to classify Iris flower species based on their sepal length, sepal width, petal length, and petal width. The neural network is built using Scikit-Learn and PyTorch and is trained on the well-known Iris dataset.

Dataset

The Iris dataset is a classic dataset in machine learning and statistics. It contains 150 samples of Iris flowers, with 50 samples each of three species: Iris setosa, Iris versicolor, and Iris virginica. Each sample has four features:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

Project Structure

- simple_NeuralNetwork.ipynb: This Jupyter notebook contains the code for loading the dataset, preprocessing the data, building and training the neural network, and evaluating its performance.

Requirements
To run this project, you need the following Python libraries:

1. numpy
2. pandas
3. scikit-learn
4. torch
5. matplotlib

You can install these dependencies using pip: pip install numpy pandas scikit-learn torch matplotlib

Usage
1. Load the Dataset: The Iris dataset is loaded using Scikit-Learn's load_iris function.

2. Preprocess the Data: The dataset is split into training and testing sets. Features are standardized for better neural network performance.

3. Build the Neural Network: A simple neural network is built using PyTorch. The network consists of an input layer, one hidden layer, and an output layer.

4. Train the Network: The network is trained using the training dataset. The loss function used is Cross-Entropy Loss, and the optimizer is Stochastic Gradient Descent (SGD).

5. Evaluate the Network: The performance of the network is evaluated on the testing dataset, and accuracy is calculated.

Example
Here is a brief example of how to load the dataset and print its first few rows:

from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows
print(df.head())

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The Iris dataset is sourced from the UCI Machine Learning Repository.
This project is inspired by various machine learning tutorials and documentation from Scikit-Learn and PyTorch.
Repository
The complete code for this project is available in the GitHub repository: Neural Network for Iris Flower Classification

