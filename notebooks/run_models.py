
'''

TO COMPARE MODEL PERFORMANCE, RUN THIS SCRIPT ONLY

The purpose of this script is to run each model via a function defined in each model's module.
These functions will return a DataFrame of predictions and targets, which can be plotted as the 
results. The trading results are also plotted using the function in src/trading.py.

'''



#from _03_random_forest import run_random_forest#
from _04_cnn import read_data, normalise_data
#from _05_FCN_ExtraTrees_XGBoost import run_hybrid
import yfinance as yf
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#import from elsewhere in the project
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from show_results import present_model_results
from trading import baseline, simulate_trading

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


sp500_data = read_data('^GSPC', start =  "2000-01-01")

#get baseline and S&P500 profit yields

def get_baseline_and_index_results(ticker, sp500_data):
    data = read_data(ticker)
    baseline_results, baseline_trade_count, baseline_wins, baseline_losses, baseline_total_roi, baseline_roi_per_trade =  baseline(data)
    baseline_outcome = [baseline_results, baseline_trade_count, baseline_wins, baseline_losses, baseline_total_roi, baseline_roi_per_trade ]
    #sp500 results
    # S&P 500 returns over the same period

    initial_capital = 10000
    sp500_data["Returns"] = sp500_data["Close"].pct_change()
    sp500_data["Portfolio_Value"] = (1 + sp500_data["Returns"]).cumprod() * initial_capital
    sp500_data['Profit'] = sp500_data['Portfolio_Value'] - initial_capital
    return baseline_results, sp500_data

## plot all the profit results from each model, compared to a (single) baseline and index fund (sp500)

def plot_trading_results(model_results, baseline, index_fund):
        model_colours = ['#2ea647', '#472ea6']
        models = ['RF', 'CNN', 'Hybrid']
        fig, ax = plt.subplots(1,1,figsize = (4,4), layout = 'constrained')
        
        for i, result in enumerate(model_results):
           # result.set_index('Date', inplace=True)
            ax.plot(baseline.index[baseline.shape[0]-result.shape[0]:], result["Profit"],c = model_colours[i], ls = 'dashed', label = f'MSFT Portfolio Profit, {models[i]}')
        ax.plot(baseline.index, baseline["Profit"], 'b--',label = 'Baseline Strategy Profit')
        ax.plot(index_fund.index, index_fund["Profit"], 'k-', label = 'Index Fund Profit')
        ax.legend()
        ax.grid()
        ax.set_title("Trading Strategy Performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("USD ($) Value")
        fig.savefig(f'trading_strategy.png', dpi=600)


# to import from the src/ folder when running the notebook

data = yf.Ticker('MSFT')
'''
model_3_preds = pd.read_csv('notebooks/model_iii_predictions.csv')
data_df = read_data('MSFT')
data_df = normalise_data(data_df)
print(model_3_preds)
model_3_preds.set_index('Date', inplace=True)
model_3_profit, a, b,c, d, e, f = simulate_trading(data_df[4222+2500:], pd.DataFrame(data = model_3_preds['Predictions'],index = data_df[4222+2500:].index))
model_3_profit.to_csv('notebooks/model_iii_profit.csv')
'''
trading_results_set = []
# get the baseline profit and sp500 profit to compare
baseline_outcome, sp500 = get_baseline_and_index_results('MSFT', sp500_data)

# take results csv if not found
y_dataframes_set = []


model_results_data_files = [f'notebooks/model_{k}_predictions.csv' for k in [ 'i', 'ii', 'iii']]
model_trading_data_files = [f'notebooks/model_{k}_profit.csv' for k in [ 'i', 'ii']]
print(model_results_data_files)
print("heppp")
for j, model_results in enumerate(model_results_data_files):
    if os.path.exists(model_results):
     y_dataframes_set.append(pd.read_csv(model_results))
for j, model_results in enumerate(model_trading_data_files):
    if os.path.exists(model_results):
     trading_results = pd.read_csv(model_trading_data_files[j])
    # trading_results.set_index('Date', inplace=True)
     trading_results_set.append(trading_results)






present_model_results([y['Target'].to_numpy() for y in y_dataframes_set], [y['Probabilities'].to_numpy() for y in y_dataframes_set])

# generate the trading results figure

plot_trading_results(trading_results_set, baseline_outcome, sp500)