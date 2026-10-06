# XENIOS – Trading Signal Terminal
Deploy: push this folder to GitHub → share.streamlit.io → New app → main file `app.py`.
Data: Yahoo Finance via yfinance (NSE = `SYMBOL.NS`). If the feed is unreachable the app falls back to clearly-labelled DEMO data instead of crashing.

Chart is a custom Plotly.js component (`chart_component/`) so zoom/pan survive live refreshes. Keep that folder in the repo.
