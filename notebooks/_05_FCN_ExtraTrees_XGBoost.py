import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from trading_simulation import run_full_simulation

#=============================================================================
# Configuration
#=============================================================================

SEEDS = [13, 42, 97]
LOOKBACK = 20
TICKER, PERIOD = "MSFT", "20y"
LOOKAHEAD = 90
EMBARGO = LOOKAHEAD
INIT_TRAIN, STEP = 2500, 90
N_SELECT, MARGIN, META_TH = 12, 0.25, 0.5
ALPHA = 0.8
THRESHOLD = 0.51
FCN_EPOCHS = 25

FEATURES = [
    #Returns & volatility
    "ret_1", "ret_5", "ret_21", "ret_63", "logret_1",
    "rv_20", "rv_63", "atr_14",
    #Trend & distances
    "dist_sma_10", "dist_sma_20", "dist_sma_50", "dist_sma_100",
    "sma_10_20", "sma_20_50", "sma_50_100",
    "ema_12_26",
    #Oscillators / momentum (Appendix A)
    "momentum_10", "roc_10",
    "stoch_k_14", "stoch_d_14",
    "williams_r_14",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    #Bands / range
    "bb_percent_b", "bb_bw",
    "adx_14",
    #Volume
    "obv_z20", "volume_z20",
]

#=============================================================================
#Features and target
#=============================================================================

def add_features(df):
    #Exponential smoothing on OHLCV
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].ewm(alpha=ALPHA, adjust=False).mean()
    
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    v = df["Volume"].astype(float)

    #Returns
    df["ret_1"] = c.pct_change()
    df["logret_1"] = np.log(c).diff()
    df["ret_5"] = c.pct_change(5)
    df["ret_21"] = c.pct_change(21)
    df["ret_63"] = c.pct_change(63)

    #Realised volatility (Appendix A, annualised)
    df["rv_20"] = df["ret_1"].rolling(20, min_periods=20).std() * np.sqrt(252)
    df["rv_63"] = df["ret_1"].rolling(63, min_periods=63).std() * np.sqrt(252)

    #SMA distances & ratios
    sma10 = c.rolling(10, min_periods=10).mean()
    sma20 = c.rolling(20, min_periods=20).mean()
    sma50 = c.rolling(50, min_periods=50).mean()
    sma100 = c.rolling(100, min_periods=100).mean()
    df["dist_sma_10"] = c / sma10 - 1
    df["dist_sma_20"] = c / sma20 - 1
    df["dist_sma_50"] = c / sma50 - 1
    df["dist_sma_100"] = c / sma100 - 1
    df["sma_10_20"] = sma10 / sma20 - 1
    df["sma_20_50"] = sma20 / sma50 - 1
    df["sma_50_100"] = sma50 / sma100 - 1

    #EMA relationship
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    df["ema_12_26"] = ema12 / ema26 - 1

    #Momentum + ROC (Appendix A)
    df["momentum_10"] = c - c.shift(9)
    df["roc_10"] = (c - c.shift(10)) / c.shift(10) * 100

    #Stochastic %K/%D and Williams %R (Appendix A)
    hh14 = h.rolling(14, min_periods=14).max()
    ll14 = l.rolling(14, min_periods=14).min()
    rng14 = (hh14 - ll14)
    stoch_k = (c - ll14) / rng14 * 100
    df["stoch_k_14"] = stoch_k
    df["stoch_d_14"] = stoch_k.rolling(3, min_periods=3).mean()
    df["williams_r_14"] = (hh14 - c) / rng14 * 100

    #MACD (Appendix A)
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    #RSI(14) (Appendix A) using Wilder smoothing
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    #Bollinger %B and bandwidth
    bb_mid = c.rolling(20, min_periods=20).mean()
    bb_std = c.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower)
    df["bb_percent_b"] = (c - bb_lower) / bb_range
    df["bb_bw"] = bb_range / bb_mid

    #ATR(14) as fraction of price (Appendix A style)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    df["atr_14"] = atr / c

    #ADX(14)
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(14, min_periods=14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=14).sum() / tr14
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=14).sum() / tr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx_14"] = dx.rolling(14, min_periods=14).mean()

    #OBV z-score (Appendix A) + volume z-score
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    df["obv_z20"] = (obv - obv.rolling(20, min_periods=20).mean()) / obv.rolling(20, min_periods=20).std()
    df["volume_z20"] = (v - v.rolling(20, min_periods=20).mean()) / v.rolling(20, min_periods=20).std()

    return df.replace([np.inf, -np.inf], np.nan)

#=============================================================================
# Target and sequences
#=============================================================================

def price_direction_labels(close, lookahead):
    fwd = close.shift(-lookahead)
    return np.where(fwd.isna(), np.nan, (fwd > close).astype(float))

def build_sequences(X, y, lookback):
    xs = [X[i - lookback + 1: i + 1] for i in range(lookback - 1, len(X))]
    return np.array(xs), y[lookback - 1:]

def stitch_eval(X_tr, X_ev, lb):
    return np.vstack([X_tr[-(lb - 1):], X_ev])

class FCN:
    def __init__(self, n_features, lookback=LOOKBACK, epochs=40, batch_size=128):
        self.lookback, self.epochs, self.batch_size = lookback, epochs, batch_size
        x = inp = tf.keras.Input(shape=(lookback, n_features))
        for f, k in [(96, 5), (192, 3), (96, 3)]:
            x = tf.keras.layers.Conv1D(f, k, padding="same",
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(tf.keras.layers.GlobalAveragePooling1D()(x))
        self.model = tf.keras.Model(inp, out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                           loss="binary_crossentropy", metrics=["accuracy"])

    def train(self, X_tr, y_tr, class_weight=None, X_val=None, y_val=None):
        Xt, yt = build_sequences(X_tr, y_tr, self.lookback)
        val = None
        if X_val is not None and y_val is not None and len(X_val) >= self.lookback:
            Xv, yv = build_sequences(X_val, y_val, self.lookback)
            val = (Xv, yv)
        kw = dict(epochs=self.epochs, batch_size=self.batch_size,
                  class_weight=class_weight, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss" if val is not None else "loss",
                      patience=8,
                      restore_best_weights=True
                  )])
        self.model.fit(Xt, yt, validation_data=val, **kw)

    def predict(self, X_ev):
        return self.model.predict(build_sequences(X_ev, np.zeros(len(X_ev)), self.lookback)[0], verbose=0).flatten()

def correlation_filter(*Xs, threshold=0.85):
    X_tr = Xs[0]
    corr, n, drop = np.corrcoef(X_tr.T), X_tr.shape[1], set()
    for i in range(n):
        if i in drop: continue
        drop |= {j for j in range(i + 1, n) if abs(corr[i, j]) > threshold}
    keep = [i for i in range(n) if i not in drop]
    return tuple(X[:, keep] for X in Xs)

#=============================================================================
# Walk forward fold
#=============================================================================

def run_fold(payload):
    fold_idx, n_folds, train_end, test_end, X_raw, y_raw, k = payload
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    tr_end = max(train_end - EMBARGO, 1)
    Xs = [X_raw[:tr_end], X_raw[train_end:test_end]]
    y_tr = y_raw[:tr_end]
    y_te = y_raw[train_end:test_end]

    Xs = list(correlation_filter(*Xs))
    sel = SelectKBest(mutual_info_classif, k=min(k, Xs[0].shape[1])).fit(Xs[0], y_tr)
    Xs = [sel.transform(x) for x in Xs]
    
    # MinMax normalization
    s = MinMaxScaler().fit(Xs[0])
    X_tr, X_te = s.transform(Xs[0]), s.transform(Xs[1])
    te_seq = stitch_eval(X_tr, X_te, LOOKBACK)

    classes = np.unique(y_tr)
    cw = ({int(c): float(v) for c, v in zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr))}
          if len(classes) == 2 else None)
    sw_tr = np.exp(np.linspace(-0.693, 0.0, len(X_tr)))
    n0 = int((y_tr == 0).sum())
    n1 = int((y_tr == 1).sum())
    spw = n0 / max(n1, 1)

    # Fast stacking split inside the training window
    split = int(len(X_tr) * 0.8)
    X_base, y_base = X_tr[:split], y_tr[:split]
    X_stack, y_stack = X_tr[split:], y_tr[split:]
    sw_base = sw_tr[:split]

    # ----------------------------
    # Base learners
    # ----------------------------
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
        tree_method="hist",
    )
    xgb.fit(
        X_base,
        y_base,
        sample_weight=sw_base,
        eval_set=[(X_stack, y_stack)],
        verbose=False,
    )

    etc = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    etc.fit(X_base, y_base)

    tf.keras.utils.set_random_seed(SEEDS[0])
    fcn = FCN(n_features=X_tr.shape[1], epochs=FCN_EPOCHS, batch_size=512)
    X_stack_seq = stitch_eval(X_base, X_stack, LOOKBACK)
    y_stack_pad = np.concatenate([np.zeros(LOOKBACK - 1, dtype=y_stack.dtype), y_stack])
    fcn.train(X_base, y_base, class_weight=cw, X_val=X_stack_seq, y_val=y_stack_pad)

    fcn_stack_p = fcn.predict(X_stack_seq)
    etc_stack_p = etc.predict_proba(X_stack)[:, 1]
    xgb_stack_p = xgb.predict_proba(X_stack)[:, 1]
    meta_X = np.column_stack([fcn_stack_p, etc_stack_p, xgb_stack_p])

    meta = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    meta.fit(meta_X, y_stack)

    fcn_tp = fcn.predict(te_seq)
    etc_tp = etc.predict_proba(X_te)[:, 1]
    xgb_tp = xgb.predict_proba(X_te)[:, 1]

    ens_probs = meta.predict_proba(np.column_stack([fcn_tp, etc_tp, xgb_tp]))[:, 1]
    
    #Pick a threshold that balances both classes on the stack split
    stack_probs = meta.predict_proba(meta_X)[:, 1]
    thr_grid = np.linspace(0.35, 0.65, 13)
    best_thr, best_score = THRESHOLD, -1.0
    for thr in thr_grid:
        p = (stack_probs > thr).astype(int)
        score = f1_score(y_stack, p, average="macro", zero_division=0)
        if score > best_score:
            best_score, best_thr = score, float(thr)

    preds = (ens_probs > best_thr).astype(int)
    meta_probs = np.where(preds == 1, ens_probs, 1.0 - ens_probs)

    return {"fold_idx": fold_idx, "n_folds": n_folds, "y_test": y_te, "preds": preds,
            "ensemble_probs": ens_probs, "meta_probs": meta_probs,
            "acc": accuracy_score(y_te, preds) * 100,
            "bal": balanced_accuracy_score(y_te, preds) * 100,
            "up_rate": y_te.mean() * 100, "fallback_margin": 0.0,
            "maj_class": int(y_tr.mean() > 0.5)}

#=============================================================================
# Reporting
#=============================================================================

def print_metrics(y, yhat):
    macro = dict(average="macro", zero_division=0)
    for label, value in [
        ("Accuracy", accuracy_score(y, yhat)),
        ("Balanced Accuracy", balanced_accuracy_score(y, yhat)),
        ("Precision", precision_score(y, yhat, **macro)),
        ("Recall", recall_score(y, yhat, **macro)),
        ("F-Measure", f1_score(y, yhat, **macro))]:
        print(f"{label}: {value * 100:.2f}%")

def print_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{title}")
    print(f"True Down: {cm[0][0]} | False Up: {cm[0][1]}")
    print(f"False Down: {cm[1][0]} | True Up: {cm[1][1]}")

def plot_results(y, yhat, probs):
    out = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{TICKER} {LOOKAHEAD}-Day Price-Direction Prediction ({PERIOD})",
                 fontsize=15, fontweight="bold", y=1.02)

    ax = axes[0]
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1])
    ax.set_xticklabels(["Pred PriceDown", "Pred PriceUp"])
    ax.set_yticklabels(["True PriceDown", "True PriceUp"])
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    fpr, tpr, _ = roc_curve(y, probs)
    ax.plot(fpr, tpr, "steelblue", linewidth=2, label=f"ROC (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    ax.set(title="ROC Curve", xlabel="FPR", ylabel="TPR", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    pr, rc, _ = precision_recall_curve(y, probs)
    ax.plot(rc, pr, "steelblue", linewidth=2, label=f"PR curve (AUC = {auc(rc, pr):.3f})")
    ax.axhline(y.mean(), color="gray", ls="--", alpha=0.7, label=f"Baseline ({y.mean():.2f})")
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.savefig(f"{out}/results_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Results analysis saved to: {out}/results_analysis.png")

def plot_feature_importance(df):
    out = os.path.dirname(os.path.abspath(__file__))
    X = df[FEATURES].values
    y = df["Direction"].values.astype(int)
    mi = mutual_info_classif(X, y, random_state=42)
    order = np.argsort(mi)
    plt.figure(figsize=(9, 6))
    plt.barh(np.array(FEATURES)[order], mi[order], color="steelblue")
    plt.title(f"{TICKER} Feature Importance for {LOOKAHEAD}-Day Price Direction")
    plt.xlabel("Mutual information")
    plt.tight_layout()
    plt.savefig(f"{out}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance saved to: {out}/feature_importance.png")

#=============================================================================
# Data and folds
#=============================================================================

def load_dataset():
    print(f"Loading {TICKER} ({PERIOD})...")
    raw = yf.download(TICKER, period=PERIOD, auto_adjust=True, progress=False)
    if raw.empty:
        print("Error: no data."); sys.exit(1)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    print("Building features...")
    #Keep the true close for target + trading
    close_trade = raw["Close"].astype(float).copy()

    #Feature engineering smooth OHLCV in-place
    df = add_features(raw.copy())
    df["Close_trade"] = close_trade.reindex(df.index)

    df["Direction"] = price_direction_labels(df["Close_trade"], LOOKAHEAD)
    return df.dropna()

def make_fold_args(df):
    y_raw, X_raw = df["Direction"].values.astype(int), df[FEATURES].values
    n = len(X_raw)
    print(f"\nFeatures ({len(FEATURES)}): {', '.join(FEATURES)}")
    k = min(N_SELECT, len(FEATURES))
    if n <= INIT_TRAIN + STEP:
        sys.exit(f"Error: not enough data (n={n}, need > {INIT_TRAIN + STEP}).")
    fold_starts = list(range(INIT_TRAIN, n, STEP))
    print(f"\nExpanding walk-forward: {len(fold_starts)} folds "
          f"(T0={INIT_TRAIN}, step={STEP}, embargo={EMBARGO}, features={len(FEATURES)}->{k})\n")
    return [(i, len(fold_starts), ts, min(ts + STEP, n), X_raw, y_raw, k) for i, ts in enumerate(fold_starts, 1)]

#=============================================================================
# Run helpers
#=============================================================================

def _print_fold(r):
    print(f"Fold {r['fold_idx']}/{r['n_folds']} acc={r['acc']:.2f}% bal={r['bal']:.2f}% "
          f"up={r['up_rate']:.1f}% m={r['fallback_margin']:.3f} maj={r['maj_class']}")
    return r

def run_walk_forward(fold_args, workers):
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [_print_fold(r) for r in pool.map(run_fold, fold_args)]
    else:
        results = [_print_fold(run_fold(a)) for a in fold_args]
    results.sort(key=lambda r: r["fold_idx"])
    return results

def aggregate_predictions(results):
    cat = lambda key: np.concatenate([r[key] for r in results])
    return cat("y_test"), cat("preds"), cat("ensemble_probs"), cat("meta_probs")

def report_results(results, all_true, all_pred, all_probs, all_meta):
    up_rate = all_true.mean() * 100
    print("\nAggregate Results:")
    print_metrics(all_true, all_pred)
    try:
        print(f"ROC-AUC: {roc_auc_score(all_true, all_probs):.4f}")
        print(f"PR-AUC: {average_precision_score(all_true, all_probs):.4f}")
    except Exception: pass
    accs = np.array([r["acc"] for r in results])
    bals = np.array([r["bal"] for r in results])
    print(f"\nPer-fold: acc {accs.mean():.2f}% +/-{accs.std():.2f}% bal {bals.mean():.2f}% +/-{bals.std():.2f}%")
    print(f"Up rate: {up_rate:.1f}% (majority baseline = {max(up_rate, 100-up_rate):.1f}%)")
    print_cm(all_true, all_pred, "Confusion Matrix (full coverage):")
    
    def section(mask, title, header):
        n_sel = int(mask.sum())
        print(f"\n{header}\nCoverage: {n_sel/len(all_true)*100:.1f}%  ({n_sel}/{len(all_true)} days)")
        if n_sel == 0: print("No predictions."); return
        print_metrics(all_true[mask], all_pred[mask])
        print_cm(all_true[mask], all_pred[mask], title)
    
    section(np.abs(all_probs - 0.5) >= MARGIN, "Confusion Matrix (selective):", f"Selective (|p-0.5| >= {MARGIN}):")
    section(all_meta >= META_TH, "Confusion Matrix (meta):", f"Meta-labelling (is-correct prob >= {META_TH}):")

#=============================================================================
# Main
#=============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--step", type=int, default=STEP, help="Walk-forward step size (days). Higher = fewer folds = faster.")
    p.add_argument("--init-train", type=int, default=INIT_TRAIN, help="Initial training window size (days).")
    p.add_argument("--n-select", type=int, default=N_SELECT, help="How many features to select per fold.")
    p.add_argument("--fcn-epochs", type=int, default=FCN_EPOCHS, help="FCN training epochs per fold.")
    p.add_argument("--trade-size", type=float, default=1000.0, help="Trade size in currency units.")
    p.add_argument("--trade-threshold", type=float, default=THRESHOLD, help="Trade signal threshold on predicted prob.")
    args = p.parse_args()
    workers = args.workers

    STEP = args.step
    INIT_TRAIN = args.init_train
    N_SELECT = min(args.n_select, len(FEATURES))
    FCN_EPOCHS = args.fcn_epochs
    trade_threshold = float(args.trade_threshold)
    
    df = load_dataset()
    fold_args = make_fold_args(df)
    results = run_walk_forward(fold_args, workers)
    all_true, all_pred, all_probs, all_meta = aggregate_predictions(results)
    report_results(results, all_true, all_pred, all_probs, all_meta)
    
    print("\nPer-class metrics:")
    for c in (1, 0):
        p = precision_score(all_true, all_pred, pos_label=c, zero_division=0)
        r = recall_score(all_true, all_pred, pos_label=c, zero_division=0)
        f1 = f1_score(all_true, all_pred, pos_label=c, zero_division=0)
        print(f"y={c}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    
    plot_results(all_true, all_pred, all_probs)
    plot_feature_importance(df)
    
    test_idx = np.concatenate([np.arange(a[2], a[3]) for a in fold_args])
    trade_pred = (all_probs > trade_threshold).astype(int)
    print(f"\nTrading signal rate (p>{trade_threshold:.3f}): {trade_pred.mean()*100:.1f}%")
    preds_df = pd.DataFrame({"Predictions": trade_pred}, index=df.index[test_idx])
    df_prices = df[["Close_trade"]].rename(columns={"Close_trade": "Close"})
    run_full_simulation(
        df_prices,
        preds_df,
        TICKER,
        PERIOD,
        horizon=90,
        trade_size=float(args.trade_size),
        train_cutoff=INIT_TRAIN,
        model_name="hybrid",
    )
    
preds_df = pd.DataFrame({
    "Target": all_true,
    "Predictions": all_pred,
    "Probabilities": all_probs
})

preds_df.to_csv("model_iii_predictions.csv", index=False)