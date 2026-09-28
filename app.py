"""Interactive dashboard:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from quantsim import (STRATEGIES, backtest_universe, bootstrap_paths, buy_and_hold, clean_prices, fetch_yfinance,
                      generate_synthetic_prices, inject_dirt, risk_report, summary)
from quantsim.viz import plot_drawdowns, plot_equity, plot_fan

st.set_page_config(page_title="QuantSim", layout="wide")
st.title("QuantSim: Backtesting & Monte Carlo Engine")

with st.sidebar:
    source = st.radio("Data source", ["Synthetic (offline)", "Yahoo Finance"])
    tickers = st.text_input("Tickers", "AAPL,MSFT,GOOGL,AMZN,JPM")
    strat = st.selectbox("Strategy", list(STRATEGIES))
    cost = st.slider("Costs (bps per trade)", 0.0, 30.0, 5.0)
    params = {}
    if strat == "sma_crossover":
        params = dict(fast=st.slider("Fast window", 5, 50, 20), slow=st.slider("Slow window", 60, 250, 100))
    elif strat == "momentum":
        params = dict(lookback=st.slider("Lookback (days)", 20, 250, 120))
    else:
        params = dict(window=st.slider("Window", 10, 60, 20), entry_z=st.slider("Entry z", 1.0, 3.0, 2.0),
                      exit_z=st.slider("Exit z", 0.0, 1.0, 0.5))
    horizon = st.slider("Monte Carlo horizon (days)", 30, 504, 252)


@st.cache_data
def load(src, tk):
    raw = (inject_dirt(generate_synthetic_prices()) if src.startswith("Synth")
           else fetch_yfinance([t.strip() for t in tk.split(",")], "2015-01-01"))
    return clean_prices(raw)


prices, rep = load(source, tickers)
with st.expander("ETL report"):
    st.json(rep.as_dict())

res = {"buy_and_hold": buy_and_hold(prices),
       strat: backtest_universe(prices, STRATEGIES[strat], name=strat, cost_bps=cost, **params)}
st.dataframe(pd.DataFrame({k: summary(v.returns) for k, v in res.items()}).T.style.format("{:.3f}"))

c1, c2, c3 = st.columns(3)
for col, fn in zip((c1, c2), (plot_equity, plot_drawdowns)):
    fig, ax = plt.subplots(); fn(res, ax); col.pyplot(fig)
paths = bootstrap_paths(res[strat].returns, horizon, 3000)
fig, ax = plt.subplots(); plot_fan(paths, ax); c3.pyplot(fig)
st.json({k: round(v, 4) for k, v in risk_report(paths).items()})
