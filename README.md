# Employee Salary Prediction Project



## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Libraries and Datasets](#libraries-and-datasets)
- [Exploratory Data Analysis and Visualization](#exploratory-data-analysis-and-visualization)
- [Creating Training and Testing Dataset](#creating-training-and-testing-dataset)
- [Training a Linear Regression Model](#training-a-linear-regression-model)
- [Evaluating Model Performance](#evaluating-model-performance)
- [Training with Amazon SageMaker](#training-with-amazon-sagemaker)
- [Deploying and Testing the Model](#deploying-and-testing-the-model)
- [Achievements](#achievements)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [License](#license)



## Project Overview



This project aims to predict employee salary based on the number of years of experience using simple linear regression. The project explores various data analysis and visualization techniques, builds a regression model using scikit-learn, and finally leverages Amazon SageMaker to train and deploy a linear learner model.



## Problem Statement



- The objective is to predict the employee salary based on the number of years of experience.
- In simple linear regression, we predict the value of one variable Y based on another variable X.
- X is the independent variable, and Y is the dependent variable.
- The relationship is linear, meaning changes in X result in proportional changes in Y.



## Install the required dependencies:
```bash
pip install -r requirements.txt
```


## Libraries and Datasets



We use libraries like seaborn, tensorflow, pandas, numpy, matplotlib, and sagemaker. The dataset used is salary.csv, which contains information on employees' years of experience and their corresponding salaries.



## Exploratory Data Analysis and Visualization



Performed EDA to understand the distribution and relationships within the dataset, including checking for null values, statistical summary, histograms, pair plots, and correlation heatmaps.



## Creating Training and Testing Dataset



Split the dataset into training and testing sets to build and evaluate the regression model.



## Training a Linear Regression Model



Used the scikit-learn library to train a linear regression model, fitting the model to the training data and evaluating its performance.



## Evaluating Model Performance



Visualized the trained model's predictions against the actual data to evaluate its performance.



## Training with Amazon SageMaker



Utilized Amazon SageMaker to train a linear learner model, uploading the data to S3, configuring the training job, and setting hyperparameters.



## Deploying and Testing the Model



Deployed the trained model using SageMaker and tested it on the test dataset to make predictions.



## Achievements



- Managed Spot Training savings: 61.5%



## Repository Structure



- data/: Contains the dataset used for the project.
- notebooks/: Jupyter notebooks with detailed steps for data preprocessing, EDA, and modeling.
- scripts/: Python scripts for data cleaning and model training.
- results/: Output files including model evaluation reports and visualizations.
- README.md: Project overview and instructions.



## Dependencies



- pandas
- numpy
- seaborn
- matplotlib
- tensorflow
- sagemaker
- boto3
- scikit-learn



## How to Run



1. Clone the repository:
```sh
git clone https://github.com/erjonb19/employee-salary-prediction.git
