"""
Project 1: Differentiated Demand for Cars
Theoretical Question 3 -- testing the Perloff-Salop model empirically

Perloff-Salop (1985) predicts a symmetric equilibrium price:
    p* = c + 1 / [ n(n-1) * INTEGRAL( F(eps)^(n-2) * f(eps)^2 d eps ) ]

i.e. price depends only on marginal cost (c) and the intensity of competition
(n, and the shape of the match-value distribution F, which we take as fixed/
structural). Crucially, price should NOT depend on a product's own observed
characteristics (weight, fuel efficiency, ...), since in the baseline P-S
model firms are otherwise symmetric and differentiation works purely through
the i.i.d., zero-correlation match value -- not through systematic vertical
quality that would show up in equilibrium pricing.

Test: regress price on cost proxies + number of competing firms (n) +
observed characteristics (we, li), and run a joint F-test of
    H0: coefficient on we = 0  AND  coefficient on li = 0
Rejecting H0 is evidence against the (baseline) Perloff-Salop model.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("cars.csv")
df["market"] = df["ye"].astype(str) + "_" + df["ma"].astype(str)

# n_jt: number of distinct competing firms in the market (proxy for n in P-S)
df["n_firms"] = df.groupby("market")["frm"].transform("nunique")

# Cost proxy: producer price index of the exporting country (avppr).
# Drop the (small number of) rows with missing/non-positive values.
df = df[df["avppr"] > 0].dropna(subset=["avppr"]).copy()

df["log_avppr"] = np.log(df["avppr"])          # cost-side control
df["log_eurpr"] = np.log(df["eurpr"])          # price in common currency (SDR)

X = sm.add_constant(df[["log_avppr", "tax", "n_firms", "we", "li"]])
y = df["log_eurpr"]

model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["market"]})
print(model.summary())

print("\nJoint test of H0: coef(we) = 0 and coef(li) = 0")
ftest = model.f_test("we = 0, li = 0")
print(ftest)
