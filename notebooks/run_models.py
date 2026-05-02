
'''

TO COMPARE MODEL PERFORMANCE, RUN THIS SCRIPT ONLY

The purpose of this script is to run each model via a function defined in each model's module.
These functions will return a DataFrame of predictions and targets, which can be plotted as the 
results. The trading results are also plotted using the function in src/trading.py.

'''



from _03_random_forest import run_random_forest
from _04_cnn import run_cnn
from _05_FCN_ExtraTrees_XGBoost import run_hybrid
import yfinance as yf
import sys
from pathlib import Path
# to import from the src/ folder when running the notebook
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from show_results import present_model_results
from trading import plot_trading_results


data = yf.Ticker('MSFT')
y_dataframes_set = []
for model_function in [run_random_forest, run_cnn, run_hybrid]:
    # y_dataframe is the predictions dataframe from the backtesting.
    y_dataframe = model_function(data)
    y_dataframes_set.append(y_dataframe)


#generate the results figure (3 sets of results)    

present_model_results(y_dataframes_set)

# generate the trading results figure

plot_trading_results(y_dataframes_set)