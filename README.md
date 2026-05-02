# CMT307 Stock Price Prediction

Group repository for the Applied Machine Learning project.

Description of each code file:



### run_models.py

Run this file to train and test **all** models against the MSFT dataset, as well as calculate the trading algorithm using the model predictions. Produces Figure 4 (classifiation performance) and Figure 5 (trading profits) in the report.

### _01_descriptive_analysis.ipynb 

Runs exploratory data analysis of the MSFT stock price data. Plots the close price distribution, the time series, correlation between Open-High-Close-Low-Volume (OHCLV) variables, daily returns, intra-day variation. Used for Figure 1 in the report.

### _02_preprocessing_and_features.ipynb

### _03_random_forest.py

Builds the Random Forest model (Model I in the report).

### _04_cnn.py

Builds the one-dimensional convolutional neural network model (Model II in the report).

### _05_FCN_ExtraTrees_XGBoost.py

Builds the stacked hybrid model, consisting of a Fully Convolutional Network (base learner) an Extremely Randomized Decision Tree learner, and an XGBoost learner. Model III in the report.


### features.py

Contains functions to calculate certain technical indicator features, used in the models. These features are outlined in Appendix A of the report.

### show_results.py

A function for generating the classification performance results plot.

### trading.py

The trading algorithm, given in Appendix B of the report (using the model's predictions to decide when to open trades) and the baseline (opens trades each day if the held capital is enough)








