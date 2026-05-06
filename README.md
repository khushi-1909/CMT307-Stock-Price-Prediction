# CMT307 Stock Price Prediction

This repository contains the code for CMT307 Applied Machine Learning project on stock price prediction. 

The task is to predict whether the MSFT closing price will inclrease over a 90-trading-dat horizon using historical OHLC data and derived technical features. The task is treated as a binary classification problem:
- `1`: closing price after 90 days is higher than the current closing price
- `0`:closing price after 90 days is lower than or equal to the current closing price

The project includes exploratory data analysis, preprocessing and feature engineering, model training, walk forward evaluation, and a simple trading algorithm

## Project Structure
```text

data/
    README.md

notebooks/
    _01_descriptive_analysis.ipynb
    _02_preprocessing_and_features.ipynb
    _03_random_forest.ipynb
    _04_cnn.py
    _05_FCN_ExtraTrees_XGBoost.ipynb
    run_models.py

src/
    features.py
    backtesting.py
    trading.py
    show_results.py
    mlstock.py
    cnn_1d.py

docs/
archive/
outputs/
requirements.txt
README.md

```

## Setup
To run the project, first clone or download the repository and then install the requied python packages from `requirements.txt`

The project was developed using Python 3.

## Dataset
The stock data is downloaded using the `yfinance` package. 

- Ticker : `MSFT`
- Frequency: Daily Trading Data
- Date Range: January 1999 to March 2026 (approx.)
- Main Variables: Open, High, Low, Close, Adjusted Close, Volume

## Target Variable
The prediction target is based on the closing price:

```text
Target = 1 if Close[t+90] > Close[t]
Target = 0 otherwise
```

The final 90 rows are removed because the future closing price is not available for those dates.

This gives a moderately imbalanced classification task because MSFT has an overall upward trend across the selected period.


## File Description

### run_models.py

Run this file to generate all results figures in the report (classification performance and trading profit). This uses the output .csv files from the Python scripts building each model.

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








>>>>>>> c66e3992f817eb70bf96f04ce7b3dd510d21fd43
