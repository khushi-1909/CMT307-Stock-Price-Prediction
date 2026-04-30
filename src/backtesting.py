import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc



def predict_90(train, test, predictors, model, engine = 'sklearn'):


    
    if engine == 'keras':
        model.fit(train[predictors], train["Target"], epochs = 5)
        probs = model.predict(test[predictors])[:,0]
    else: 
        model.fit(train[predictors], train["Target"])
        probs = model.predict_proba(test[predictors])[:,1]
    probs = pd.Series(probs, index=test.index) # Convert numpy array to pandas Series
    preds = (probs > 0.51).astype(int)
    preds = pd.Series(preds, index=test.index)
    combined = pd.concat({"Target": test["Target"],"Predictions": preds, 'Probabilities': probs}, axis=1)
    return combined

def backtest_90(data, model, predictors, start=2500, step=180, engine = 'sklearn'):
    data = data.copy()
    all_predictions = []
    oob_scores = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict_90(train, test, predictors, model, engine = engine)

        all_predictions.append(predictions)

        if hasattr(model, "oob_score_"):
            oob_scores.append(model.oob_score_)

    return pd.concat(all_predictions), oob_scores