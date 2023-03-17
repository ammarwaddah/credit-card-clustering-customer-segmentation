# Credit Card Clustering Customer Segmentation
Using Machine Learning and Clustering techniques to organizing similar objects into groups using significant features given by the most linked features that are taken into consideration when evaluating the target.

## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Evaluation](#evaluation)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)
* [Sources](#sources)

## Introduction

Business leadership takes its place in the applications of artificial intelligence, and artificial intelligence seeks, as usual, to gain its effectiveness with it (as it seeks to enhance its effectiveness in all other fields), but the difficulty of these areas lies in in processing large and diverse data, and it's difficult to separate it into similar groups aimed at identifying their diversity, and here lies the importance of clustering in applications of artificial intelligence, thus Clustering is the act of organizing similar objects into groups, and used to identify groups of similar objects in datasets with two or more variable quantities. Based on this introduction, I will present to you my project solving the problem of Clustering, using a lot of effective algorithm and techniques with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning.\
Hoping to improve it gradually in the coming times.\

## Dataset General info

**General info about the dataset:**

`CUST_ID` Identification of Credit Card holder (Categorical).\
`BALANCE` Balance amount left in their account to make purchases.\
`BALANCE_FREQUENCY` How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated).\
`PURCHASES` Amount of purchases made from account.\
`ONEOFF_PURCHASES` Maximum purchase amount done in one-go.\
`INSTALLMENTS_PURCHASES` Amount of purchase done in installment.\
`CASH_ADVANCE` Cash in advance given by the user.\
`PURCHASES_FREQUENCY` How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased).\
`ONEOFFPURCHASESFREQUENCY` How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased).\
`PURCHASESINSTALLMENTSFREQUENCY` How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done).\
`CASHADVANCEFREQUENCY` How frequently the cash in advance being paid.\
`CASHADVANCETRX` Number of Transactions made with "Cash in Advanced".\
`PURCHASES_TRX` Numbe of purchase transactions made.\
`CREDIT_LIMIT` Limit of Credit Card for user.\
`PAYMENTS` Amount of Payment done by user.\
`MINIMUM_PAYMENTS` Minimum amount of payments made by user.\
`PRCFULLPAYMENT` Percent of full payment paid by user.\
`TENURE` Tenure of credit card service for user.
    
## Evaluation

I measure the accuracy of each algorithm using:
Inertia, Silhouette, Silhouette score diagram (Silhouette Visualizer), also (Bayesian information criterion, Distance between Gaussian mixtures) for Gaussian mixture model.

## Technologies

* Programming language: Python.
* Libraries: Numpy, Matplotlib, Pandas, Seaborn, plotly, tabulate, math, sklearn, scipy, yellowbrick. 
* Application: Jupyter Notebook.

## Setup

To run this project setup the following libraries on your local machine using pip on the terminal after installing Python:\
'''\
pip install numpy\
pip install matplotlib\
pip install pandas\
pip install seaborn\
pip install plotly\
pip install tabulate\
pip install python-math\
pip install scikit-learn\
pip install scipy\
pip install yellowbrick\

'''\
To install these packages with conda run:\
'''

conda install -c anaconda numpy\
conda install -c conda-forge matplotlib\
conda install -c anaconda pandas\
conda install -c anaconda seaborn\
conda install -c plotly plotly\
conda install -c conda-forge tabulate\
conda install -c conda-forge python-markdown-math\
conda install -c anaconda scikit-learn\
conda install -c anaconda scipy\
conda install -c districtdatalabs yellowbrick\
'''

## Features

* I will present to you my project solving the problem of Clustering (Credit Card Clustering Customer Segmentation), using a lot of effective algorithm and techniques (KMeans, Mini-Batch K-Means, DBSCAN, Agglomerative, Birch, Mean Shift, Affinity Propagation, Spectral, Gaussian mixture, and PCA using cosine similarity of data points) with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning.

### To Do:

**Briefly about the process of the project work, here are (some) insights that I took care of it:**

* Explore the dataset

* EDA step

* Preprocess step:

Dealing with missing and duplicated values after some check the effect of each value on the other feature, Check the correlation with other columns, and the distribution of each feature that has missing values.

* Check outliers:
Dealing with outliers thow Convert to z-score after process some of high probability outlier values by moving it to the nearest point in the curve (threshold values)

**Modeling:**

* I used several clustering algorithm like:
KMeans, Mini-Batch K-Means, DBSCAN, Agglomerative, Birch, Mean Shift, Affinity Propagation, Spectral, Gaussian mixture, and PCA using cosine similarity of data points.

* I measure the accuracy of each algorithm using:
Inertia, Silhouette, Silhouette score diagram (Silhouette Visualizer), also (Bayesian information criterion, Distance between Gaussian mixtures) for Gaussian mixture model.

* I used PCA with 95 percentage of the retrieved data to transform data and fit it to each model, and then visualize it using the new axes in 2D space.

## Run Example
To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage.

2. Load the dataset.

3. Select which cell you would like to run and view its output.

5. Run the rest of cells to end up in the training process and visualizing results.

4. Run Selection/Line in Python Terminal command (Shift+Enter).

## Sources
This data was taken from:
https://www.kaggle.com/arjunbhasin2013/ccdata
