
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

def plot_trading_results(results, baseline, index_fund, model_name):

        plt.figure(figsize=(10, 6))
        plt.plot(results.index, results["Profit"], 'r--', label = 'MSFT Portfolio Profit')
        plt.plot(baseline.index, baseline["Profit"], 'b--',label = 'Baseline Strategy Profit')
        plt.plot(index_fund.index, index_fund["Profit"], 'k-', label = 'Index Fund Profit')
        plt.legend()
        plt.title("Trading Strategy Performance")
        plt.xlabel("Date")
        plt.ylabel("USD ($) Value")
        plt.grid()
        plt.savefig(f'{model_name} trading_strategy.png', dpi=600)

def baseline(df, horizon=90,
                            initial_capital=10000,
                            trade_size=1000):

        df = df.copy()
        df = df.iloc[2500:]

        capital = initial_capital
        capital_history = []
        portfolio_history = []

        open_trades = []

        trade_count = 0
        wins = 0
        losses = 0

        trade_returns = []

        for i in range(len(df)):
            today_price = df["Close"].iloc[i]

            # --- CLOSE trades ---
            new_open_trades = []
            for trade in open_trades:
                if i == trade["exit_idx"]:
                # print("help")
                    entry_price = trade["entry_price"]
                    invest = trade["invest"]

                    ret = (today_price - entry_price) / entry_price
                    profit = invest * ret

                    capital += invest + profit
                    trade_returns.append(ret)

                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                else:
                    new_open_trades.append(trade)

            open_trades = new_open_trades

            # --- OPEN trade EVERY DAY if possible ---
            if i < len(df) - horizon and capital >= trade_size:
                capital -= trade_size

                open_trades.append({
                    "entry_idx": i,
                    "entry_price": today_price,
                    "exit_idx": i + horizon,
                    "invest": trade_size
                })

                trade_count += 1

            # --- PORTFOLIO VALUE ---
            open_value = sum(
                trade["invest"] * (today_price / trade["entry_price"])
                for trade in open_trades
            )

            portfolio_value = capital + open_value

            capital_history.append(capital)
            portfolio_history.append(portfolio_value)

        df["Capital"] = capital_history
        df["Portfolio_Value"] = portfolio_history
        df['Profit'] = df['Portfolio_Value'] - initial_capital

        # --- FINAL RETURN ---
        final_value = portfolio_history[-1]
        return_pct = (final_value - initial_capital) / initial_capital * 100

        # --- AVG TRADE RETURN ---
        avg_trade_return = (
            (sum(trade_returns) / len(trade_returns)) * 100
            if trade_returns else 0
        )

        return df, trade_count, wins, losses, return_pct, avg_trade_return

def simulate_trading(df, predictions, horizon=90,
                        initial_capital=10000,
                        trade_size=1000):

        df = df.copy()
        df = df.loc[predictions.index]
        print(df)
        df["Predictions"] = predictions["Predictions"]

        capital = initial_capital
        capital_history = []
        portfolio_history = []

        open_trades = []

        trade_count = 0
        wins = 0
        losses = 0

        trade_returns = []  # <-- store returns per trade

        for i in range(len(df)):
            today_price = df["Close"].iloc[i]

            # --- CLOSE trades ---
            new_open_trades = []
            for trade in open_trades:
                if i == trade["exit_idx"]:
                    entry_price = trade["entry_price"]
                    invest = trade["invest"]

                    ret = (today_price - entry_price) / entry_price
                    profit = invest * ret

                    capital += invest + profit

                    trade_returns.append(ret)  # <-- store return

                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                else:
                    new_open_trades.append(trade)

            open_trades = new_open_trades

            # --- OPEN new trade ---
            if (
                i < len(df) - horizon and
                df["Predictions"].iloc[i] == 1 and
                capital >= trade_size
            ):
                capital -= trade_size

                open_trades.append({
                    "entry_idx": i,
                    "entry_price": today_price,
                    "exit_idx": i + horizon,
                    "invest": trade_size
                })

                trade_count += 1

            # --- PORTFOLIO VALUE ---
            open_value = sum(
                trade["invest"] * (today_price / trade["entry_price"])
                for trade in open_trades
            )

            portfolio_value = capital + open_value

            capital_history.append(capital)
            portfolio_history.append(portfolio_value)

        df["Capital"] = capital_history
        df["Portfolio_Value"] = portfolio_history
        df["Profit"] = df["Portfolio_Value"] - initial_capital

        # --- FINAL RETURN % ---
        final_value = portfolio_history[-1]
        return_pct = (final_value - initial_capital) / initial_capital * 100

        # --- AVG RETURN PER TRADE (%) ---
        avg_return_per_trade = (
            (sum(trade_returns) / len(trade_returns)) * 100
            if trade_returns else 0
        )

        # Return volatility
        return_vol = ret.std()

        return df, trade_count, wins, losses, return_pct, avg_return_per_trade, return_vol
