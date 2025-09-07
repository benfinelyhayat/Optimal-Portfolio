# Optimal-Portfolio
Using MPT to create a portfolio; this is for port opt demonstration so will use basic methods for sample mean and sd but can easily incorporate methods such as rolling means and such to the model with minimal tweaking.
this is just a demonstration so will use pseudo stats. 
## N risky assets, with return given.
https://quantmathreadme.blogspot.com/2025/08/optimal-portfolio-theory.html

```
import numpy as np
individual_sd_matrix = np.array([
    [0.01, 0, 0, 0],
    [0, 0.2, 0, 0],
    [0 ,0 ,0.2 ,0],
    [0, 0, 0, 0.3]
])
Corr_Matrix = np.array([
    [0, 0.4, 0.3, 0.3],
    [0.4, 0, 0.2, 0.4],
    [0.3, 0.2, 0, 0.5],
    [0.3, 0.4, 0.5, 0]
]) + np.identity(4)

Expected_matrix = np.array([
    [0.05],
    [0.07],
    [0.1],
    [0.2]
])
Sigma = individual_sd_matrix @ Corr_Matrix @ individual_sd_matrix

columb_1s = np.array([
    [1],
    [1],
    [1],
    [1]
])
A = columb_1s.T @ np.linalg.inv(Sigma) @ columb_1s
B = Expected_matrix.T @ np.linalg.inv(Sigma) @ columb_1s
C = Expected_matrix.T @ np.linalg.inv(Sigma) @ Expected_matrix

alphebet_matrix = np.linalg.inv(np.array([
    [B.item() , A.item() ],
    [C.item() , B.item() ]
]))
idk = np.array([
    [1],
    [0.07]
])
parameter_solutions =  alphebet_matrix @ idk

w_star = parameter_solutions[0,0].item() * np.linalg.inv(Sigma) @ Expected_matrix + parameter_solutions[1,0].item() * np.linalg.inv(Sigma) @ columb_1s

w_star_renormalised = w_star /w_star.flatten().sum() #floating point error occurs without

print(w_star_renormalised)
```
here we have a code that selects the best 5 stock combinations from a list given that I cant share so all values of w are positive

```
import yfinance as yf
import pandas as pd
import itertools
import numpy as np

# List of 40 UK stock tickers (from LSE). Suffix ".L" is used for London-listed stocks on Yahoo Finance.
tickers = [
    # Renewables & Clean Energy (10)
     "UKW.L", "DRX.L", "BSIF.L", "ITM.L", "CWR.L", "TRIG.L", "ORIT.L", "JLEN.L",
    # Supermarkets & Retail (8)
    "TSCO.L", "SBRY.L", "OCDO.L", "MKS.L", "ABF.L", "KGF.L", "BME.L", "SUPR.L",
    # Pharma / Med (10)
    "AZN.L", "GSK.L", "HLN.L", "HIK.L", "INDV.L", "SN.L", "CTEC.L", "OXB.L", "GNS.L", "PRTC.L",
    # Other large/liquid LSE names
    "LGEN.L", "NG.L", "BA.L", "RIO.L", "BP.L", "HSBA.L", "BARC.L", "LLOY.L", "BATS.L"
]

# Download adjusted close prices for last 100 trading days
data = yf.download(tickers, period="150d", interval="1d", auto_adjust=False)["Adj Close"].dropna().iloc[-100:]

# Compute daily returns
returns = data.pct_change().dropna()

# Mean daily returns
mean_returns = returns.mean()

# Variance of each stock's returns
variances = returns.var()

# Covariance matrix (40x40)
cov_matrix = returns.cov()

# Display results
print("Mean Returns (daily):\n", mean_returns)
print("\nVariances (daily):\n", variances)
print("\nCovariance Matrix (40x40):\n", cov_matrix)

best_var = float("inf")
best_combo = None
best_weights = None

for i,j,k,l,m in itertools.combinations(range(len(tickers)),5):
    mew_selected = mean_returns.iloc[[i,j,k,l,m]].to_numpy().reshape(-1,1)
    sigma_selected = cov_matrix.iloc[[i,j,k,l,m], [i,j,k,l,m]].to_numpy()
    columb_1s = np.ones((5,1))
    A = columb_1s.T @ np.linalg.inv(sigma_selected) @ columb_1s
    B = mew_selected.T @ np.linalg.inv(sigma_selected) @ columb_1s
    C = mew_selected.T @ np.linalg.inv(sigma_selected) @ mew_selected
    alphebet_matrix = np.linalg.inv(np.array([
    [B.item() , A.item() ],
    [C.item() , B.item() ]
    ]))
    idk = np.array([
    [1],
    [0.0003]
    ])
    parameter_solutions =  alphebet_matrix @ idk

    w_star = parameter_solutions[0,0].item() * np.linalg.inv(sigma_selected) @ mew_selected + parameter_solutions[1,0].item() * np.linalg.inv(sigma_selected) @ columb_1s

    w_star_renormalised = w_star /w_star.flatten().sum()
    Var_portfolio = ((w_star_renormalised.T @ sigma_selected @ w_star_renormalised).item()) 
    if Var_portfolio < best_var and np.all(w_star_renormalised >= 0):
        best_var = Var_portfolio
        best_weights = w_star_renormalised
        best_combo = [tickers[x] for x in [i, j, k, l, m]]

print(best_combo, best_var, best_weights)
```
