# %% [markdown]
# ## NOTE
# model 3 is originally predicting 10 volatility
# i changed it to match the project target, but the results are not great
# 
# possible improvement: change the features?

# %%
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


# %%
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve,
                             roc_auc_score, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier


# %%
#from trading_simulation import run_full_simulation

#=============================================================================
# Configuration
#=============================================================================

SEEDS = [13, 42, 97]
LOOKBACK = 20
TICKER= "MSFT"
START_DATE = '1999-01-01' 
END_DATE = '2026-03-31'
LOOKAHEAD = 90
EMBARGO = LOOKAHEAD
TRAIN_WIN, VAL_WIN, TEST_WIN = 1260, 252, 252
N_SELECT, MARGIN, META_TH = 15, 0.2, 0.5

FEATURES = [
    "Vol_Regime", "Vol_Trend_20", "Abs_Return_Max_20", "Vol_Percentile_60d",
    "GK_20d", "EWMA_Vol_20", "RV_Ratio_5_20", "Vol_of_Vol",
    "Return_Skew_20", "Return_Kurt_20",
    "Dist_252d_High",
    "VIX_Percentile_60d", "VIX_ZScore_20d", "Vol_Risk_Premium",
    "Vol_AutoCorr_20"]


# %%
#=============================================================================
# Feature building
#=============================================================================

#Builds the features list
def add_features(df):
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]
    ret = c.pct_change()
    abs_ret = ret.abs()
    std20 = ret.rolling(20).std()
    rv5 = ret.rolling(5).std()
    log_hl = np.log(h / l).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(c / o).replace([np.inf, -np.inf], np.nan)
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

    df["GK_20d"] = np.sqrt(gk.rolling(20).mean().clip(lower=0))
    df["EWMA_Vol_20"] = ret.ewm(span=20, min_periods=10).std()
    df["RV_Ratio_5_20"] = rv5 / std20.replace(0, np.nan)
    df["Vol_of_Vol"] = std20.rolling(20).std()
    df["Vol_Regime"] = std20.rolling(252, min_periods=60).rank(pct=True)
    df["Vol_Percentile_60d"] = std20.rolling(60, min_periods=20).rank(pct=True)
    df["Vol_Trend_20"] = std20 - std20.shift(20)
    df["Vol_AutoCorr_20"] = abs_ret.rolling(20).corr(abs_ret.shift(1))
    df["Return_Skew_20"] = ret.rolling(20).skew()
    df["Return_Kurt_20"] = ret.rolling(20).kurt()
    df["Abs_Return_Max_20"] = abs_ret.rolling(20).max()
    df["Dist_252d_High"] = (c - c.rolling(252, min_periods=60).max()) / c
    return df

# %%
#Download VIX and turn it into the three columns the model needs.
def get_market_data(START_DATE, END_DATE):
    d = yf.download("^VIX", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    vix = d["Close"]
    ctx = pd.DataFrame(index=vix.index)
    ctx["VIX_Level"] = vix
    ctx["VIX_ZScore_20d"] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()
    ctx["VIX_Percentile_60d"] = vix.rolling(60, min_periods=20).rank(pct=True)
    return ctx


# Label is 1 if the close price is higher after 90 trading days, otherwise 0.
def price_direction_labels(close, horizon=LOOKAHEAD):
    future_close = close.shift(-horizon)
    return np.where(
        future_close.isna(),
        np.nan,
        (future_close > close).astype(float)
    )

# %%
#=============================================================================
# Sequence helpers and FCN model
#=============================================================================

#Cut X into overlapping 20-day windows for the Conv1D input.
def build_sequences(X, y, lookback):
    xs = [X[i - lookback + 1: i + 1] for i in range(lookback - 1, len(X))]
    return np.array(xs), y[lookback - 1:]

#Add the tail of the training data to the start of the validation or test
#set so the first window has real history instead of NaNs.
def stitch_eval(X_tr, X_ev, lb):
    return np.vstack([X_tr[-(lb - 1):], X_ev])


# %%
#Convolutional neural network with three Conv1D blocks and a sigmoid output.
class FCN:
    def __init__(self, n_features, lookback=LOOKBACK, epochs=60, batch_size=64):
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

    def train(self, X_tr, y_tr, X_val, y_val, class_weight=None):
        Xt, yt = build_sequences(X_tr, y_tr, self.lookback)
        Xv, yv = build_sequences(stitch_eval(X_tr, X_val, self.lookback),
                                 np.concatenate([np.zeros(self.lookback - 1), y_val]), self.lookback)
        self.model.fit(Xt, yt, validation_data=(Xv, yv), epochs=self.epochs,
                       batch_size=self.batch_size, class_weight=class_weight, verbose=0,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)])

    def predict(self, X_ev):
        return self.model.predict(build_sequences(X_ev, np.zeros(len(X_ev)), self.lookback)[0], verbose=0).flatten()


# %%
#=============================================================================
# Ensemble plumbing
#=============================================================================

#Build inputs for the meta-model: each base model's prediction plus measures
#of how much they agree.
def meta_features(*ps):
    from itertools import combinations
    arr = np.array(ps)
    ens = arr.mean(axis=0)
    pairs = [np.abs(a - b) for a, b in combinations(ps, 2)]
    return np.column_stack([*arr, ens, np.abs(ens - 0.5), arr.std(axis=0), *pairs])


# %%
#Drop features that are too similar to ones we have already kept.
def correlation_filter(X_tr, X_val, X_te, threshold=0.85):
    corr, n, drop = np.corrcoef(X_tr.T), X_tr.shape[1], set()
    for i in range(n):
        if i in drop:
            continue
        drop |= {j for j in range(i + 1, n) if abs(corr[i, j]) > threshold}
    keep = [i for i in range(n) if i not in drop]
    return X_tr[:, keep], X_val[:, keep], X_te[:, keep]


# %%
#Calibrate predicted probabilities so they line up with real frequencies.
def calibrate(val_p, test_p, y_val):
    cal = IsotonicRegression(out_of_bounds="clip").fit(val_p, y_val)
    return cal.transform(val_p), cal.transform(test_p)


# %%
#=============================================================================
# Walk-forward fold worker
#=============================================================================

#Train all three models on one fold and return their predictions and accuracy.
def run_fold(payload):
    fold_idx, n_folds, tr_start, val_start, te_start, te_end, X_raw, y_raw, k = payload
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    #Leave a gap between train, validation, and test sets so future labels
    #cannot leak back into earlier splits.
    sl = [(0, max(tr_start + 1, val_start - EMBARGO)),
          (val_start, max(val_start + 1, te_start - EMBARGO)),
          (te_start, te_end)]
    Xs = [X_raw[a:b] for a, b in sl]
    y_tr, y_val, y_te = (y_raw[a:b] for a, b in sl)

    #Drop similar features, then keep the top k by mutual information.
    Xs = list(correlation_filter(*Xs))
    sel = SelectKBest(mutual_info_classif, k=min(k, Xs[0].shape[1])).fit(Xs[0], y_tr)
    Xs = [sel.transform(x) for x in Xs]
    s = StandardScaler().fit(Xs[0])
    X_tr, X_val, X_te = s.transform(Xs[0]), s.transform(Xs[1]), s.transform(Xs[2])
    val_seq, te_seq = stitch_eval(X_tr, X_val, LOOKBACK), stitch_eval(X_tr, X_te, LOOKBACK)

    classes = np.unique(y_tr)
    cw = ({int(c): float(v) for c, v in zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr))}
          if len(classes) == 2 else None)
    #Weight recent training rows more than older ones.
    sw_tr = np.exp(np.linspace(-0.693, 0.0, len(X_tr)))

    spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    es = dict(eval_set=[(X_val, y_val)])
    xgb_common = dict(scale_pos_weight=spw, eval_metric="logloss",
                      early_stopping_rounds=20, verbosity=0, n_jobs=2)

    def fit_pred(m, sw=None, **fit_kw):
        m.fit(X_tr, y_tr, **({"sample_weight": sw} if sw is not None else {}),
              **({"verbose": False, **fit_kw} if fit_kw else {}))
        return m.predict_proba(X_val)[:, 1], m.predict_proba(X_te)[:, 1]

    #Model 1: XGBoost (gradient-boosted trees).
    xgb_vp, xgb_tp = fit_pred(XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                            reg_lambda=1.0, random_state=42, **xgb_common),
                              sw=sw_tr, **es)

    #Model 2: FCN (neural network), averaged over three random seeds.
    fcn_vs, fcn_ts = [], []
    for seed in SEEDS:
        tf.keras.utils.set_random_seed(seed)
        fcn = FCN(n_features=X_tr.shape[1])
        fcn.train(X_tr, y_tr, X_val, y_val, class_weight=cw)
        fcn_vs.append(fcn.predict(val_seq))
        fcn_ts.append(fcn.predict(te_seq))
    fcn_vp, fcn_tp = np.mean(fcn_vs, axis=0), np.mean(fcn_ts, axis=0)

    #Model 3: ExtraTrees (random forest variant).
    etc_vp, etc_tp = fit_pred(ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                                   min_samples_leaf=20, class_weight="balanced",
                                                   random_state=42, n_jobs=2))

    #Calibrate each model's probabilities before combining them.
    raw = [(fcn_vp, fcn_tp), (etc_vp, etc_tp), (xgb_vp, xgb_tp)]
    cal = [calibrate(vp, tp, y_val) for vp, tp in raw]
    val_ps, test_ps = [v for v, _ in cal], [t for _, t in cal]

    #Blend the three models, giving more weight to whichever scored best on validation.
    two_class = len(np.unique(y_val)) == 2
    if two_class:
        aucs = np.array([roc_auc_score(y_val, p) for p in val_ps])
        aucs = np.where(aucs < 0.5, 1 - aucs, aucs)
    else:
        aucs = np.full(len(val_ps), 0.5)
    w = (e := np.clip(aucs - 0.5, 0.01, None)) / e.sum()

    def blend(ps):
        a = np.clip(np.asarray(ps), 1e-4, 1 - 1e-4)
        return 1 / (1 + np.exp(-np.dot(w, np.log(a / (1 - a)))))
    logit_val, logit_test = blend(val_ps), blend(test_ps)

    def to_logit(ps):
        a = np.clip(np.asarray(ps).T, 1e-4, 1 - 1e-4)
        return np.log(a / (1 - a))
    #Train a logistic regression on top of the three predictions and average it
    #50/50 with the blend above.
    if two_class:
        stk = LogisticRegression(C=1.0, penalty="l2", max_iter=1000,
                                 class_weight="balanced", random_state=42)
        Xv, Xt = to_logit(val_ps), to_logit(test_ps)
        stk.fit(Xv, y_val)
        sv, st = stk.predict_proba(Xv)[:, 1], stk.predict_proba(Xt)[:, 1]
        val_ens, ens_probs = 0.5 * logit_val + 0.5 * sv, 0.5 * logit_test + 0.5 * st
    else:
        val_ens, ens_probs = logit_val, logit_test

    #Find the best probability cutoff using the validation set.
    thr_grid = np.linspace(0.3, 0.7, 41)
    val_scores = [
        (t, balanced_accuracy_score(y_val, (val_ens > t).astype(int)))
        for t in thr_grid
    ]
    best_thr = max(val_scores, key=lambda x: x[1])[0]
    #If the model is too unsure (close to 50%) fall back to the majority class
    #instead of guessing. Find the best margin on validation.
    maj_class = int(y_tr.mean() > 0.5)
    val_model = (val_ens > best_thr).astype(int)
    best_m, best_acc = 0.0, (val_model == y_val).mean()
    for m in np.linspace(0.0, 0.30, 31):
        mask = np.abs(val_ens - 0.5) >= m
        p = val_model.copy()
        p[~mask] = maj_class
        acc_m = balanced_accuracy_score(y_val, p)
        if acc_m > best_acc + 1e-6:
            best_acc, best_m = acc_m, m

    #Apply the chosen cutoff and margin to the test set.
    preds = (ens_probs > best_thr).astype(int)
    preds[np.abs(ens_probs - 0.5) < best_m] = maj_class
    val_corr = (val_model == y_val).astype(int)

    #Train a second model that predicts whether each prediction is correct.
    meta_Xv, meta_Xt = meta_features(*val_ps), meta_features(*test_ps)
    if len(np.unique(val_corr)) == 2:
        meta = LogisticRegression(max_iter=1000, class_weight="balanced").fit(meta_Xv, val_corr)
        meta_probs = meta.predict_proba(meta_Xt)[:, 1]
    else:
        meta_probs = np.full(len(meta_Xt), float(val_corr.mean()) if len(val_corr) else 0.5)

    return {"fold_idx": fold_idx, "n_folds": n_folds, "y_test": y_te, "preds": preds,
            "ensemble_probs": ens_probs, "meta_probs": meta_probs,
            "acc": accuracy_score(y_te, preds) * 100,
            "bal": balanced_accuracy_score(y_te, preds) * 100,
            "up_rate": y_te.mean() * 100, "fallback_margin": best_m, "maj_class": maj_class}

# %%
#=============================================================================
# Reporting helpers
#=============================================================================

#Print accuracy, precision, recall and F1 as a clean table.
def print_metrics(y, yhat):
    macro = dict(average="macro", zero_division=0)
    for label, fn, kw in [("Accuracy",          accuracy_score,          {}),
                          ("Balanced Accuracy", balanced_accuracy_score, {}),
                          ("Precision",         precision_score,         macro),
                          ("Recall",            recall_score,            macro),
                          ("F-Measure",         f1_score,                macro)]:
        print(f"{label+':':<18} {fn(y, yhat, **kw)*100:.2f}%")


# %%
#Print the confusion matrix as a 2x2 grid.
def print_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{title}\nTrue Down: {cm[0][0]} | False Up:   {cm[0][1]}\nFalse Down:{cm[1][0]} | True Up:    {cm[1][1]}")

# %%
#=============================================================================
# Results dashboard
#=============================================================================

#Save a three-panel results figure: confusion matrix, ROC, PR curve.
def plot_results(results, y, yhat, probs):
    out = os.getcwd()
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
    f"{TICKER} 90-Day Price Direction Prediction",
    fontsize=15,
    fontweight="bold",
    y=1.02 )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual",
           xticks=[0, 1], yticks=[0, 1])
    ax.set_xticklabels(["Pred Down", "Pred Up"])
    ax.set_yticklabels(["True Down", "True Up"])
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y, probs)
    ax.plot(fpr, tpr, "steelblue", linewidth=2, label=f"ROC (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    ax.set(title="ROC Curve", xlabel="FPR", ylabel="TPR", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    pr, rc, _ = precision_recall_curve(y, probs)
    ax.plot(rc, pr, "steelblue", linewidth=2, label=f"PR curve (AUC = {auc(rc, pr):.3f})")
    ax.axhline(y.mean(), color="gray", ls="--", alpha=0.7, label=f"Baseline ({y.mean():.2f})")
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.savefig(f"{out}/results_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Results analysis saved to: {out}/results_analysis.png")


# %%
#=============================================================================
# Training stages
#=============================================================================

#Download MSFT data, build all features, add VIX columns and the 90-day price direction label.
def load_dataset():
    print(f"Loading {TICKER} from {START_DATE} to {END_DATE}...")
    raw = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
    )    
    if raw.empty:
        print("Error: no data.")
        sys.exit(1)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    print("Building features...")
    df = add_features(raw.copy()).join(get_market_data(START_DATE, END_DATE), how="left")
    #Variance Risk Premium = VIX divided by realised volatility.
    df["Vol_Risk_Premium"] = (df["VIX_Level"] / 100) / (df["EWMA_Vol_20"] * np.sqrt(252)).replace(0, np.nan)
    df["Direction"] = price_direction_labels(df["Close"], LOOKAHEAD)    
    return df.dropna()

# %%
#Build the list of fold arguments that get passed to run_fold.
def make_fold_args(df):
    y_raw, X_raw = df["Direction"].values.astype(int), df[FEATURES].values
    n = len(X_raw)
    print(f"\nFeatures ({len(FEATURES)}): {', '.join(FEATURES)}")

    tw, vw, sw = TRAIN_WIN, VAL_WIN, TEST_WIN
    k = min(N_SELECT, len(FEATURES))
    if n < tw + sw:
        tw = max(int(n * 0.55), 60)
        vw, sw = max(int(tw * 0.2), 20), max(n - tw, 40)
        print(f"Short dataset: train={tw-vw}d val={vw}d test={sw}d")
    if vw >= tw or n < tw + sw:
        sys.exit(f"Error: bad split (n={n}, train={tw}, val={vw}, test={sw}).")

    #One fold per non-overlapping test window.
    fold_starts = list(range(tw, n - sw + 1, sw))
    print(f"\nWalk-forward: {len(fold_starts)} folds "
          f"(train={tw-vw}d val={vw}d test={sw}d features={len(FEATURES)}->{k})\n")

    return [(i, len(fold_starts), ts - tw, ts - vw, ts, ts + sw, X_raw, y_raw, k)
            for i, ts in enumerate(fold_starts, 1)]


# %%
#Print a one-line summary for each completed fold.
def _print_fold(r):
    print(f"Fold {r['fold_idx']:2d}/{r['n_folds']}  acc={r['acc']:5.2f}%  bal={r['bal']:5.2f}%  "
          f"up={r['up_rate']:4.1f}%  m={r['fallback_margin']:.3f}  maj={r['maj_class']}")
    return r


# %%
#Run every fold (in parallel if workers > 1) and return them in order.
def run_walk_forward(fold_args, workers):
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [_print_fold(r) for r in pool.map(run_fold, fold_args)]
    else:
        results = [_print_fold(run_fold(a)) for a in fold_args]
    results.sort(key=lambda r: r["fold_idx"])
    return results


# %%
#Stitch the per-fold predictions into one long array.
def aggregate_predictions(results):
    cat = lambda key: np.concatenate([r[key] for r in results])
    return cat("y_test"), cat("preds"), cat("ensemble_probs"), cat("meta_probs")

#Print the full, selective, and meta-labelled accuracy reports.
def report_results(results, all_true, all_pred, all_probs, all_meta):
    up_rate = all_true.mean() * 100
    print("\nAggregate Results:")
    print_metrics(all_true, all_pred)
    try:
        print(f"ROC-AUC:           {roc_auc_score(all_true, all_probs):.4f}")
        print(f"PR-AUC:            {average_precision_score(all_true, all_probs):.4f}")
    except Exception:
        pass
    accs = np.array([r["acc"] for r in results])
    bals = np.array([r["bal"] for r in results])
    print(f"\nPer-fold:  acc {accs.mean():.2f}% +/-{accs.std():.2f}%  "
          f"bal {bals.mean():.2f}% +/-{bals.std():.2f}%")
    print(f"Up rate:   {up_rate:.1f}%  (majority baseline = {max(up_rate, 100-up_rate):.1f}%)")
    print_cm(all_true, all_pred, "Confusion Matrix (full coverage):")

    def section(mask, title, header):
        n_sel = int(mask.sum())
        print(f"\n{header}\nCoverage: {n_sel/len(all_true)*100:.1f}%  ({n_sel}/{len(all_true)} days)")
        if n_sel == 0:
            print("No predictions.")
            return
        print_metrics(all_true[mask], all_pred[mask])
        print_cm(all_true[mask], all_pred[mask], title)

    #Selective: only count predictions where the model is confident.
    section(np.abs(all_probs - 0.5) >= MARGIN,
            "Confusion Matrix (selective):", f"Selective (|p-0.5| >= {MARGIN}):")
    #Meta: only count predictions the meta-model thinks are correct.
    section(all_meta >= META_TH,
            "Confusion Matrix (meta):", f"Meta-labelling (is-correct prob >= {META_TH}):")


# %%
#=============================================================================
# Run notebook version
#=============================================================================

workers = 1   # keep this as 1 inside Jupyter/VS Code notebooks

df = load_dataset()
fold_args = make_fold_args(df)

results = run_walk_forward(fold_args, workers)

all_true, all_pred, all_probs, all_meta = aggregate_predictions(results)

report_results(results, all_true, all_pred, all_probs, all_meta)
plot_results(results, all_true, all_pred, all_probs)

# Save predictions for later trading simulation / comparison.
test_idx = np.concatenate([np.arange(a[4], a[5]) for a in fold_args])

preds_df = pd.DataFrame(
    {
        "True": all_true,
        "Predictions": all_pred,
        "Probability": all_probs
    },
    index=df.index[test_idx]
)

out = os.getcwd()
preds_df.to_csv(f"{out}/model_iii_predictions.csv")

print(f"Predictions saved to: {out}/model_iii_predictions.csv")


