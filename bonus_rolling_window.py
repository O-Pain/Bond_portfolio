import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pd.set_option("display.width", 200, "display.max_columns", None)

# 0. Same data as atelier_02_pca.py: monthly changes in bp of US Treasury yields
ids = ["DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
y = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(ids), index_col=0, parse_dates=True, na_values=".").loc["2006-02-09":]
y.columns = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
dy = y.resample("ME").last().diff().dropna() * 100
win = 60  # rolling window: 5 years of monthly changes
pal = ["#2a78d6", "#eb6834", "#1baf7a"]


def fit(d):
    """PCA on a window -> explained variance and hedge ratios (in DV01). Ratios do not depend on the arbitrary PC signs."""
    p = PCA(3).fit(d)
    L = pd.DataFrame(p.components_.T, index=d.columns)
    wings = np.linalg.solve(L.loc[["2Y", "10Y"], [0, 1]].T, L.loc["5Y", [0, 1]])  # 2Y/10Y DV01 that offset PC1 and PC2 of 1 DV01 of 5Y
    return pd.Series([*p.explained_variance_ratio_, L.loc["2Y", 0] / L.loc["10Y", 0], *wings],
                     index=["PC1", "PC2", "PC3", "steep 10Y", "fly 2Y", "fly 10Y"])


# 1. Full-sample estimate (uses the future), expanding (all months before t) and rolling (the 60 months before t)
full = fit(dy)
expd = pd.DataFrame({dy.index[t]: fit(dy.iloc[:t]) for t in range(win, len(dy))}).T
roll = pd.DataFrame({dy.index[t]: fit(dy.iloc[t - win:t]) for t in range(win, len(dy))}).T
print(pd.concat([full.rename("full sample"), roll.describe().T[["min", "max"]]], axis=1).round(2))

# 2. Instability through time: solid = rolling 60m, dotted = expanding, dashed = full sample
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
for ax, cols, loc in zip(axs, [["PC1", "PC2", "PC3"], ["steep 10Y", "fly 2Y", "fly 10Y"]], ["center left", "upper left"]):
    for c, col in zip(cols, pal):
        ax.plot(roll[c], color=col, label=c)
        ax.plot(expd[c], color=col, ls=":", lw=1)
        ax.axhline(full[c], color=col, ls="--", lw=1)
    ax.legend(loc=loc, fontsize=9)
axs[0].set_title("Explained variance share, rolling 60-month PCA", fontsize=10)
axs[1].set_title("Hedge ratios in DV01 per 1 DV01 of 2Y (steepener) or 5Y (fly)", fontsize=10)
plt.tight_layout()
plt.savefig("bonus_rolling.png", dpi=150)
plt.show()

# 3. Loadings estimated on three regimes; signs aligned on the full sample so the curves are comparable
Lfull = PCA(3).fit(dy).components_.T
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
for end, col in zip(["2015-12", "2019-12", "2024-12"], pal):
    L = PCA(3).fit(dy.loc[:end].iloc[-win:]).components_.T
    L = L * np.sign((L * Lfull).sum(axis=0))  # flip a PC when it points the other way than the full-sample one
    for i, ax in enumerate(axs):
        ax.plot(dy.columns, L[:, i], marker="o", color=col, label=f"5y to {end}")
for i, ax in enumerate(axs):
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(["PC1 (level)", "PC2 (slope)", "PC3 (curvature)"][i], fontsize=10)
axs[0].legend(fontsize=9)
plt.tight_layout()
plt.savefig("bonus_loadings_regimes.png", dpi=150)
plt.show()


# 4. Out-of-sample test on the same months: which weights leave the least level (and slope) risk in the trade?
def r2(pnl, *x):
    """Share of the trade's variance still explained by the factor moves x (0 = perfectly hedged against them)"""
    X = np.column_stack([np.ones(len(pnl)), *x])
    res = pnl - X @ np.linalg.lstsq(X, pnl, rcond=None)[0]
    return 1 - res.var() / pnl.var()


d = dy.loc[roll.index]
lvl, slope = d.mean(axis=1), d["10Y"] - d["2Y"]  # model-free level and slope moves
steep = {"DV01-neutral 1:1": d["2Y"] - d["10Y"],
         "PC1-neutral, full sample (look-ahead)": d["2Y"] - full["steep 10Y"] * d["10Y"],
         "PC1-neutral, expanding": d["2Y"] - expd["steep 10Y"] * d["10Y"],
         "PC1-neutral, rolling 60m": d["2Y"] - roll["steep 10Y"] * d["10Y"]}
fly = {"DV01 50/50": d["5Y"] - 0.5 * d["2Y"] - 0.5 * d["10Y"],
       "PC1/PC2-neutral, full sample (look-ahead)": d["5Y"] - full["fly 2Y"] * d["2Y"] - full["fly 10Y"] * d["10Y"],
       "PC1/PC2-neutral, expanding": d["5Y"] - expd["fly 2Y"] * d["2Y"] - expd["fly 10Y"] * d["10Y"],
       "PC1/PC2-neutral, rolling 60m": d["5Y"] - roll["fly 2Y"] * d["2Y"] - roll["fly 10Y"] * d["10Y"]}
periods = {"2011-2026": slice(None), "ZLB 2011-2015": slice(None, "2015"), "Since 2022": slice("2022", None)}
print("R2 of the steepener on level moves\n", pd.DataFrame({p: {k: r2(v[s], lvl[s]) for k, v in steep.items()} for p, s in periods.items()}).round(3))
print("R2 of the fly on level + slope moves\n", pd.DataFrame({p: {k: r2(v[s], lvl[s], slope[s]) for k, v in fly.items()} for p, s in periods.items()}).round(3))
