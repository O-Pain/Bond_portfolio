import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pd.set_option("display.width", 200, "display.max_columns", None)  # print wide tables in full

# 1. Import US Treasury yields from FRED (no API key needed with the CSV endpoint)
ids = ["DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(ids)
y = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".").loc["2006-02-09":]  # DGS30 resumes on 2006-02-09
y.columns = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
print(y.tail())

# 2. Month-end yields -> monthly changes in basis points
ym = y.resample("ME").last()
dy = ym.diff().dropna() * 100
print(dy.describe().round(1))

# 3. PCA on the covariance matrix (sklearn centers the data but does not scale it)
pca = PCA().fit(dy)
loadings = pd.DataFrame(pca.components_.T, index=dy.columns, columns=[f"PC{i+1}" for i in range(len(ids))])
scores = pd.DataFrame(pca.transform(dy), index=dy.index, columns=loadings.columns)
print(loadings.iloc[:, :3].round(2))

# 4. Explained variance: individual and cumulative share per component
ev = pd.DataFrame({"share": pca.explained_variance_ratio_, "cumul": pca.explained_variance_ratio_.cumsum()}, index=loadings.columns)
print((ev.head(5) * 100).round(1))

# 5. Loadings of the first three PCs against maturity in years (log scale so the short end is readable)
mat = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ax = loadings.iloc[:, :3].set_index(pd.Index(mat)).plot(marker="o", color=pal[:3], figsize=(8, 4.5))
ax.axhline(0, color="grey", lw=0.8)
ax.set(xscale="log", xticks=mat, xticklabels=loadings.index, xlabel="Maturity", ylabel="Loading", title="Loadings of the first three PCs – monthly UST changes, 2006–2026")
plt.tight_layout()
plt.savefig("loadings_pc123.png", dpi=150)
plt.show()

# 7a. Yield curve at a few key dates (month-end)
dates = pd.to_datetime(["2019-08-31", "2020-03-31", "2021-12-31", "2022-12-31", "2023-06-30", ym.index[-1]])
ax = ym.loc[dates].T.set_index(pd.Index(mat)).rename(columns=lambda d: d.strftime("%b %Y")).plot(marker="o", color=pal, figsize=(8, 4.5))
ax.set(xscale="log", xticks=mat, xticklabels=loadings.index, xlabel="Maturity", ylabel="Yield (%)", title="US Treasury curve at key dates")
plt.tight_layout()
plt.savefig("curves_key_dates.png", dpi=150)
plt.show()

# 7b. Cumulated PC scores (≈ level of each factor over time), months with an inverted 2s10s shaded in grey
inv = (ym["10Y"] - ym["2Y"]) < 0
cum = scores.iloc[:, :3].cumsum()
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
for ax, c, name, col in zip(axs, cum, ["level", "slope", "curvature"], pal):
    ax.plot(cum[c], color=col)
    ax.fill_between(inv.index, 0, 1, where=inv, transform=ax.get_xaxis_transform(), color="grey", alpha=0.25, lw=0)
    ax.set_title(f"{c} ({name}) – cumulated score (bp), grey = inverted 2s10s", fontsize=10)
plt.tight_layout()
plt.savefig("scores_cumulated.png", dpi=150)
plt.show()

# 7c. Episodes: sum of PC scores vs actual yield changes (bp) over each window
eps = {"Covid (Feb-Mar 2020)": ("2020-02", "2020-03"), "Inflation shock (2022)": ("2022-01", "2022-12"), "Inversion deepens (Jul 2022-Jun 2023)": ("2022-07", "2023-06"), "Disinversion (Jul 2023-Sep 2024)": ("2023-07", "2024-09")}
tab = pd.DataFrame({k: pd.concat([scores.loc[a:b].iloc[:, :3].sum(), dy.loc[a:b, ["3M", "2Y", "10Y", "30Y"]].sum()]) for k, (a, b) in eps.items()}).T
tab["2s10s"] = tab["10Y"] - tab["2Y"]
print(tab.round(0))

# 8. Factor exposures of simple trades; weights are DV01s (positive = long bond), exposure = weights @ loadings
L = loadings.iloc[:, :3]
wings = np.linalg.solve(L.loc[["2Y", "10Y"], ["PC1", "PC2"]].T, L.loc["5Y", ["PC1", "PC2"]])  # hedge PC1 and PC2 of the 5Y belly
W = pd.DataFrame({"long 5Y (duration)": {"5Y": 1},
                  "2s10s steepener DV01-neutral": {"2Y": 1, "10Y": -1},
                  "2s10s steepener PC1-neutral": {"2Y": 1, "10Y": -L.loc["2Y", "PC1"] / L.loc["10Y", "PC1"]},
                  "2s5s10s fly PC1/PC2-neutral": {"2Y": -wings[0], "5Y": 1, "10Y": -wings[1]}}).fillna(0)
expo = W.T @ L.loc[W.index]
print(W.round(2), expo.round(3), sep="\n")
print((expo * np.sqrt(pca.explained_variance_[:3])).round(1))  # 1-sigma monthly yield move per factor, bp per unit of DV01
