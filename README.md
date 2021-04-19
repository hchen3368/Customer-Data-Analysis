

# Customer-Data-Analysis

This repository documents a data science project. In the project, we analyze and model a credit card customer dataset to answer business questions.




### Table of Contents


1. [Overview](#motivation)
2. [File Descriptions](#files)
3. [Results](#results)
4. [Environment](#installation)
5. [Data Source](#source)
5. [Licensing](#licensing)



## Overview<a name="motivation"></a>

 Understanding your customers is the key to successful business decision making. In this project, we analyze and model a credit card customer dataset to explore the following three questions:

(1). **Customer Churn Prediction**, can we predict which customers will stop using our product in the future?

(2). **Customer Segmentation**, is there a clustering structure in the dataset?

(3).  **Customer Lifetime Value**, how to better estimate each customer's lifetime value?



## File Descriptions <a name="files"></a>

- 1_Notebook: Data Cleaning and EDA.


- 2_Notebook: Churn Prediction.


- 3_Notebook: Customer Segementation.


- 4_Notebook: Customer Lifetime Value Analysis.

- 0_BankChurners.csv: This is the raw data.

- 1_data_after_Notebook1: A pickle file that caches the data processed in 1_Notebook for later use.

- helper_functions: A python file contains several utility functions.

## Results<a name="results"></a>

- To question (1): Model that predict customer churn with a 94% recall score on test data. Details in 2_Notebook.

- To question (2): Yes! There are interesting clustering structures on the dataset (K-Mean Clustering with k=2 and k=3). Details in 3_Notebook.

- To question (3): A comprehensive analysis on customer lifetime value, incorporating churn probabilty predicted in 2_Notebook. Details in 4_Notebook.


The main findings of this project will be summarized in a post [here (currently under construction)](https://tba).

## Environment <a name="installation"></a>

- The Anaconda distribution of Python3.
- Jupyter Notebook.  
- XGBoost. See [here](https://xgboost.readthedocs.io/en/latest/build.html) for installation guide.

## Data Source <a name="source"></a>

The dataset comes from to this [Kaggle Dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers).

## Licensing <a name="licensing"></a>

 You can find the Licensing for the dataset and other descriptive information in the link above.  Other than that, feel free to use anything here as you would like!
