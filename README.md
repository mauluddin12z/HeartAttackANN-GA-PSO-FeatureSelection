# Heart Attack Classification with ANN, GA, and PSO for Feature Selection

This project compares Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) for feature selection in heart attack classification using an Artificial Neural Network (ANN). The dataset used is sourced from Kaggle.

## Introduction

Predicting heart attacks is crucial for timely intervention. This project utilizes machine learning techniques, specifically ANN with GA and PSO, to accurately classify instances of heart attacks based on selected features.

## Methodology

### Data Collection

The dataset used is sourced from Kaggle and contains relevant features for heart attack prediction, such as age, sex, cholesterol levels, etc.

### Data Preprocessing

- Data cleaning to handle missing values.
- Encoding categorical variables.
- Feature scaling to standardize the data.

### Model Building

- Artificial Neural Network (ANN) is employed for building the predictive model.

### Feature Selection

Two methods are compared for feature selection:

1. **Genetic Algorithm (GA)**:
   - GA is used to optimize feature selection by evolving a population of solutions to find the best subset of features that maximize the model performance.
2. **Particle Swarm Optimization (PSO)**:
   - PSO is applied to explore the feature space and find an optimal subset of features that maximize the fitness function (model performance).

### Model Evaluation

- The performance of the ANN model with GA-selected features is compared with the performance using PSO-selected features.
- Evaluation metrics such as accuracy, precision, recall, and F1-score are used for comparison.

## Dependencies

Ensure you have the following dependencies installed:

- Python (>=3.10)
- numpy
- pandas
- scikit-learn
- PyGAD (for Genetic Algorithm)
- pyswarm (for PSO)



1. **Data Preparation**:
   - Download the heart attack dataset from Kaggle.
   - Preprocess the dataset (cleaning, encoding, scaling).
2. **Model Training and Evaluation**:
   - Run the Python script for training the ANN model with GA and PSO feature selection.
   - Compare the performance of the model using GA-selected features versus PSO-selected features.

## Acknowledgements

- Kaggle for providing the heart attack dataset.
- PyGAD library for implementing Genetic Algorithm.
- pyswarm library for implementing Particle Swarm Optimization.
