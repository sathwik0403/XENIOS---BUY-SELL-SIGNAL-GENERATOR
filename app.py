import html, datetime as dt
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, streamlit as st
import plotly.graph_objects as go, yfinance as yf

st.set_page_config(page_title="XENIOS", page_icon="📈", layout="wide")
IST = ZoneInfo("Asia/Kolkata")
G, R = "#00e676", "#ff5252"
OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
TF = {"1m": ("1m", "5d"), "5m": ("5m", "30d"), "15m": ("15m", "60d"),
      "30m": ("30m", "60d"), "4h": ("1h", "730d"), "1D": ("1d", "max")}
ASSETS = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW JONES": "^DJI", "NIKKEI 225": "^N225",
          "HANG SENG": "^HSI", "GIFT NIFTY*": "^NSEI", "BRENT CRUDE": "BZ=F", "USD/INR": "USDINR=X"}
US = ("^GSPC", "^IXIC", "^DJI")
SECT = {"S&P 500": "IT, Banks, Broad market", "NASDAQ": "IT, Tech, Midcap growth",
        "DOW JONES": "Industrials, Banks", "NIKKEI 225": "Auto, Electronics, Capital goods",
        "HANG SENG": "Metals, Commodities, Banks", "GIFT NIFTY*": "Nifty heavyweights",
        "BRENT CRUDE": "OMCs, Paints, Aviation, Upstream oil", "USD/INR": "IT & Pharma (+) / Oil importers (-)"}

st.markdown("""<style>
.stApp{background:#000}.block-container{padding-top:1rem;max-width:1700px}
.xen{font-size:2.4rem;font-weight:800;letter-spacing:.5rem;text-align:center;color:#fff;margin:0}
.sec{color:#9aa0a6;font-size:.8rem;letter-spacing:.15rem;font-weight:700;border-bottom:1px solid #222;
padding-bottom:4px;margin:6px 0 8px}
.panel{border:1px solid #222;border-radius:10px;padding:10px 12px;background:#0a0a0a}
.tbl{width:100%;border-collapse:collapse;font-size:.82rem}.tbl th{color:#9aa0a6;text-align:left;
border-bottom:1px solid #333;padding:5px}.tbl td{padding:6px 5px;border-bottom:1px solid #161616;vertical-align:top}
.badge{display:inline-block;border:1px solid;border-radius:4px;padding:2px 8px;font-weight:700;font-size:.8rem;white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.tile{border-radius:6px;padding:10px 6px;text-align:center;color:#fff;font-size:.82rem;font-weight:600}
.ltp{font-size:2rem;font-weight:700;line-height:1}.small{color:#9aa0a6;font-size:.75rem}
div[data-testid="stButton"] button{background:#0b2a17;border:1px solid #00e676;color:#00e676;font-weight:700}
</style>""", unsafe_allow_html=True)

# ---------------- data ----------------
def resolve(q):
    q = (q or "").strip().upper().replace(" ", "")
    m = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
    if not q: return "RELIANCE.NS"
    if q in m: return m[q]
    return q if any(c in q for c in "^.=-") else q + ".NS"

def is_nse(s): return s.endswith((".NS", ".BO")) or s in ("^NSEI", "^NSEBANK", "^BSESN")

def clean(df):
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df[~df.index.duplicated()]

def resample4h(df):
    return df.resample("4h", origin="start_day", offset="9h15min").agg(OHLC).dropna()

@st.cache_data(ttl=3, show_spinner=False)
def _h_fast(sym, iv, per): return clean(yf.Ticker(sym).history(period=per, interval=iv))
@st.cache_data(ttl=30, show_spinner=False)
def _h_slow(sym, iv, per): return clean(yf.Ticker(sym).history(period=per, interval=iv))

def synth(tf):
    n = 400
    freq = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "4h": "4h", "1D": "1D"}[tf]
    idx = pd.date_range(end=pd.Timestamp.now(tz=IST).floor("min"), periods=n, freq=freq)
    rng = np.random.default_rng(7)
    c = 250 * np.exp(np.cumsum(rng.normal(0, .004 if tf in ("4h", "1D") else .0012, n)))
    c = c * 250 / c[-1]; o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, .001, n))); l = np.minimum(o, c) * (1 - abs(rng.normal(0, .001, n)))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)

def get_ohlc(sym, tf):
    c = st.session_state.get("csv")
    if c is not None:
        if tf == "1D": return c.resample("1D").agg(OHLC).dropna(), False
        if tf == "4h": return resample4h(c), False
        return c, False
    iv, per = TF[tf]
    try:
        df = (_h_fast if tf in ("1m", "5m") else _h_slow)(sym, iv, per)
        if tf == "4h" and not df.empty: df = resample4h(df)
        if len(df) > 5: return df, False
    except Exception: pass
    return synth(tf), True

def load_csv(f):
    d = pd.read_csv(f); d.columns = [c.strip().lower() for c in d.columns]
    t = d[next(c for c in d.columns if c in ("time", "date", "datetime", "timestamp"))]
    idx = pd.to_datetime(t, unit="s", utc=True) if pd.api.types.is_numeric_dtype(t) else pd.to_datetime(t, utc=True)
    d.index = pd.DatetimeIndex(idx).tz_convert(IST)
    d = d[["open", "high", "low", "close"]].astype(float)
    d.columns = ["Open", "High", "Low", "Close"]
    return d.sort_index()

@st.cache_data(ttl=20, show_spinner=False)
def snap(tickers):
    res = {}
    try:
        df = yf.download(list(tickers), period="7d", interval="1d", group_by="ticker",
                         progress=False, threads=True, auto_adjust=False)
    except Exception: return res
    for t in tickers:
        try:
            s = (df[t]["Close"] if len(tickers) > 1 else df["Close"]).dropna()
            res[t] = (float(s.iloc[-1]), float((s.iloc[-1] / s.iloc[-2] - 1) * 100), s.index[-1])
        except Exception: pass
    return res

@st.cache_data(ttl=3600, show_spinner=False)
def corrs(sym, tickers):
    try:
        ts = [t for t in tickers if t != sym]
        raw = yf.download(ts + [sym], period="1y", interval="1d", group_by="ticker", progress=False, threads=True)
        cl = pd.DataFrame({t: raw[t]["Close"] for t in ts + [sym]})
        if getattr(cl.index, "tz", None) is not None: cl.index = cl.index.tz_localize(None)
        r = cl.pct_change()
        for t in ts:
            if t in US: r[t] = r[t].shift(1)
        return {t: float(r[t].corr(r[sym])) for t in ts}
    except Exception: return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fundamentals(sym):
    try: return yf.Ticker(sym).info or {}
    except Exception: return {}

@st.cache_data(ttl=300, show_spinner=False)
def news(q):
    try:
        r = requests.get("https://news.google.com/rss/search", timeout=6, headers={"User-Agent": "Mozilla/5.0"},
                         params={"q": q + " stock", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        return [(i.findtext("title"), i.findtext("link"), i.findtext("pubDate"), i.findtext("source") or "")
                for i in ET.fromstring(r.content).iter("item")][:8]
    except Exception: return []

@st.cache_data(ttl=20, show_spinner=False)
def vix():
    try:
        t = yf.Ticker("^INDIAVIX"); h = t.history(period="5d", interval="5m")
        if h.empty: h = t.history(period="5d", interval="1d")
        return float(h.Close.iloc[-1]), h.index[-1].tz_convert(IST)
    except Exception: return None

# ---------------- analysis ----------------
def swings(df, k=2):
    h, l = df.High.values, df.Low.values; H, L = [], []
    for i in range(k, len(df) - k):
        if h[i] == h[i - k:i + k + 1].max(): H.append((i, h[i]))
        if l[i] == l[i - k:i + k + 1].min(): L.append((i, l[i]))
    return H, L

def trend_fib(h4):
    H, L = swings(h4.tail(300))
    if len(H) < 2 or len(L) < 2: return "SIDEWAYS", None
    if H[-1][1] > H[-2][1] and L[-1][1] > L[-2][1]:
        lows = [x for x in L if x[0] < H[-1][0]]
        return "UPTREND", ((lows[-1][1], H[-1][1], "up") if lows else None)
    if H[-1][1] < H[-2][1] and L[-1][1] < L[-2][1]:
        highs = [x for x in H if x[0] < L[-1][0]]
        return "DOWNTREND", ((L[-1][1], highs[-1][1], "down") if highs else None)
    return "SIDEWAYS", None

FIB_UP = {0.382: "NORMAL BUY", 0.5: "MODERATE BUY", 0.618: "STRONG BUY", 0.786: "MODERATE SELL"}
FIB_DN = {0.382: "NORMAL SELL", 0.5: "MODERATE SELL", 0.618: "STRONG SELL", 0.786: "MODERATE BUY"}
FIB_ATH = {0.382: "SUFFICIENT BUY", 0.5: "MODERATE BUY", 0.618: "STRONG BUY"}

def fib_price(leg, p):
    lo, hi, dr = leg
    return hi - p * (hi - lo) if dr == "up" else lo + p * (hi - lo)

def psy_step(p): return 1 if p < 20 else 5 if p < 100 else 50 if p < 1000 else 100 if p < 10000 else 500

def strength(a): return "WEAK" if a < .6 else "NORMAL" if a < 1 else "MODERATE" if a < 1.5 else "GOOD" if a < 2.5 else "STRONG"

def gen_signals(sym):
    now = dt.datetime.now(IST).strftime("%I:%M %p").lstrip("0"); out = []
    ltp = float(get_ohlc(sym, "1m")[0].Close.iloc[-1])
    h4 = get_ohlc(sym, "4h")[0]; d = get_ohlc(sym, "1D")[0]
    trend, leg = trend_fib(h4)
    def add(sig, price, why): out.append(dict(time=now, signal=sig, price=price, reason=why))
    tol = ltp * .003
    if leg:
        for p, sig in (FIB_UP if leg[2] == "up" else FIB_DN).items():
            lv = fib_price(leg, p)
            if abs(ltp - lv) <= tol: add(sig, lv, f"{p*100:.1f}% Fib retracement on 4H ({trend.lower()})")
    if trend == "DOWNTREND":
        cl = d.Close.copy(); cl.iloc[-1] = ltp
        for n, sig in ((21, "NORMAL SELL"), (50, "GOOD SELL"), (200, "STRONG SELL")):
            v = cl.rolling(n).mean().iloc[-1]
            if pd.notna(v) and abs(ltp - v) <= ltp * .006:
                add(sig, float(v), f"Price near daily {n} SMA in downtrend – bearish bounce likely")
        ath, atl = d.High.max(), d.Low.min()
        for p, sig in FIB_ATH.items():
            lv = ath - p * (ath - atl)
            if abs(ltp - lv) <= ltp * .004: add(sig, lv, f"{p*100:.1f}% Fib level from ATH {ath:,.2f} (downtrend)")
    step = psy_step(ltp)
    vic = max(.75 * (d.High - d.Low).tail(20).mean(), ltp * .0035)   # data-driven vicinity: 0.75x avg daily range
    if trend == "UPTREND":
        lv = np.ceil(ltp / step) * step
        if 0 <= lv - ltp <= vic: add("STRONG BUY", lv, f"Price {ltp:,.2f} approaching psychological level {lv:,.0f} (uptrend magnet)")
    elif trend == "DOWNTREND":
        lv = np.floor(ltp / step) * step
        if 0 <= ltp - lv <= vic: add("STRONG SELL", lv, f"Price {ltp:,.2f} falling to psychological level {lv:,.0f} (downtrend magnet)")
    try:
        sn = snap(tuple(ASSETS.values())); cr = corrs(sym, tuple(ASSETS.values()))
        for name, t in ASSETS.items():
            if t not in sn or t not in cr or t == sym or pd.isna(cr[t]) or abs(cr[t]) < .25: continue
            pct = sn[t][1]; a = abs(pct) * (2.5 if name == "USD/INR" else 1)
            if a < .3: continue
            buy = np.sign(pct) * np.sign(cr[t]) > 0
            add(f"{strength(a)} {'BUY' if buy else 'SELL'}", None, f"{name} {pct:+.2f}% · 6M correlation {cr[t]:+.2f} with stock")
    except Exception: pass
    return out, trend

def do_gen():
    try:
        s, t = gen_signals(resolve(st.session_state.q))
        st.session_state.signals, st.session_state.trend = s, t
    except Exception:
        st.session_state.signals, st.session_state.trend = [], "N/A"
    st.session_state.sig_sym = resolve(st.session_state.q)

def orb_info(sym):
    m, _ = get_ohlc(sym, "5m"); day = m.index[-1].date()
    s = m[[i.date() == day for i in m.index]]
    if pd.Timestamp.now(tz=s.index.tz) - s.index[-1] < pd.Timedelta("5min"): s = s.iloc[:-1]
    if len(s) < 3: return {"status": "WAIT"}
    hi, lo = s.High.iloc[:3].max(), s.Low.iloc[:3].min(); st_ = "INSIDE"
    for c in s.Close.iloc[3:]:
        if c > hi: st_ = "ABOVE"; break
        if c < lo: st_ = "BELOW"; break
    return dict(status=st_, hi=hi, lo=lo, start=s.index[0])

def momentum(d):
    c = d.Close; out = []
    for a, b, na, nb in ((21, 50, 21, 50), (50, 200, 50, 200)):
        diff = (c.rolling(a).mean() - c.rolling(b).mean()).dropna().tail(10)
        if len(diff) > 1:
            if diff.iloc[0] < 0 < diff.iloc[-1]: out.append(f"{na} SMA crossed ABOVE {nb} SMA recently – possible bullish change in momentum")
            elif diff.iloc[0] > 0 > diff.iloc[-1]: out.append(f"{na} SMA crossed BELOW {nb} SMA recently – possible bearish change in momentum")
    return out or ["No recent SMA crossover – momentum unchanged"]

def sma_line(df, d, tf, n):
    s = d.Close.rolling(n).mean()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize(); s = s[~s.index.duplicated()]
    dates = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return pd.Series((s if tf == "1D" else s.shift(1)).reindex(dates).values, index=df.index)

def badge(sig):
    c = G if "BUY" in sig else R
    return f'<span class="badge" style="color:{c};border-color:{c}">{html.escape(sig)}</span>'

def tile(name, pct, val=None):
    if pct is None: return f'<div class="tile" style="background:#222">{name}<br>—</div>'
    bg = "#0e6b35" if pct >= 0 else "#8e1c1c"
    v = f"<br>{val:,.2f}" if val is not None else ""
    return f'<div class="tile" style="background:{bg}">{name}{v}<br><b>{pct:+.2f}%</b></div>'

# ---------------- state / header ----------------
st.session_state.setdefault("csv", None)
st.session_state.setdefault("q", "RELIANCE")
st.markdown('<p class="xen">XENIOS</p>', unsafe_allow_html=True)

@st.fragment(run_every=3)
def ltp_block():
    sym = resolve(st.session_state.q); df, demo = get_ohlc(sym, "1m"); d = get_ohlc(sym, "1D")[0]
    ltp = float(df.Close.iloc[-1])
    prev = float(d.Close.iloc[-2] if len(d) > 1 and d.index[-1].date() == df.index[-1].date() else d.Close.iloc[-1])
    pct = (ltp / prev - 1) * 100; c = G if pct >= 0 else R
    tag = '<span style="color:#ffb300">● DEMO DATA (live feed unavailable)</span>' if demo else \
          f'<span style="color:{G}">● LIVE</span> · last tick {df.index[-1].strftime("%d %b %H:%M")}'
    st.markdown(f'<div class="small">LTP</div><span class="ltp">{ltp:,.2f}</span> '
                f'<span style="color:{c};font-weight:700;font-size:1.1rem">{"▲" if pct>=0 else "▼"} {pct:+.2f}%</span>'
                f'<div class="small">{tag}</div>', unsafe_allow_html=True)

a, b = st.columns([1.3, 3])
with a: ltp_block()
with b: st.text_input("Search", key="q", label_visibility="collapsed",
                      placeholder="🔍 Search NSE symbol – e.g. TCS, SBIN, INFY, NIFTY (BSE: add .BO)")
with st.expander("⬆ IMPORT TRADINGVIEW CSV (optional)"):
    f = st.file_uploader("CSV with time/open/high/low/close", type="csv", key="csvfile")
    if f is not None:
        try: st.session_state.csv = load_csv(f); st.success(f"Loaded {len(st.session_state.csv)} candles")
        except Exception as e: st.session_state.csv = None; st.error(f"Could not parse CSV: {e}")
    else: st.session_state.csv = None
sym = resolve(st.session_state.q)
r1, r2, r3 = st.columns([1.2, 3, 2.5])
r1.markdown(f'<div class="sec" style="font-size:1.1rem;color:#fff">{sym}</div>', unsafe_allow_html=True)
r2.radio("tf", list(TF), index=1, horizontal=True, key="tf", label_visibility="collapsed")
with r3:
    k1, k2, k3 = st.columns(3)
    k1.checkbox("SMA 21/50/200", True, key="t_sma"); k2.checkbox("FIB", True, key="t_fib"); k3.checkbox("ORB", True, key="t_orb")
if st.session_state.get("sig_sym") != sym: do_gen()

# ---------------- chart ----------------
@st.fragment(run_every=3)
def chart_block():
    sym = resolve(st.session_state.q); tf = st.session_state.tf
    df, demo = get_ohlc(sym, tf); d = get_ohlc(sym, "1D")[0]
    fig = go.Figure(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name=sym,
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"), showlegend=False))
    if st.session_state.t_sma:
        for n, col in ((21, "#f5c518"), (50, "#2196f3"), (200, "#e040fb")):
            fig.add_trace(go.Scatter(x=df.index, y=sma_line(df, d, tf, n), name=f"SMA {n}", mode="lines",
                                     line=dict(color=col, width=1.4)))
    if st.session_state.t_fib:
        trend, leg = trend_fib(get_ohlc(sym, "4h")[0])
        if leg:
            for p in (0.236, 0.382, 0.5, 0.618, 0.786):
                fig.add_hline(y=fib_price(leg, p), line_dash="dot", line_width=.8, line_color="#888",
                              annotation_text=f"{p*100:.1f}%", annotation_font_color="#888", annotation_position="bottom left")
    if st.session_state.t_orb and tf in ("1m", "5m", "15m", "30m"):
        try:
            o = orb_info(sym)
            if o["status"] != "WAIT":
                for y, t in ((o["hi"], "OR HIGH"), (o["lo"], "OR LOW")):
                    fig.add_shape(type="line", x0=o["start"], x1=df.index[-1], y0=y, y1=y, line=dict(color="#ffb300", width=1.2, dash="dash"))
                    fig.add_annotation(x=df.index[-1], y=y, text=t, showarrow=False, font=dict(color="#ffb300", size=10), xanchor="left")
        except Exception: pass
    for s in st.session_state.get("signals", []):
        if s["price"]:
            c = G if "BUY" in s["signal"] else R
            fig.add_hline(y=s["price"], line_dash="dot", line_width=1, line_color=c, annotation_text=s["signal"],
                          annotation_font_color=c, annotation_position="top left")
    vis = {"1m": 150, "5m": 150, "15m": 140, "30m": 120, "4h": 120, "1D": 150}[tf]; sl = df.tail(vis)
    pad = (sl.High.max() - sl.Low.min()) * .08 or 1
    step = (sl.index[-1] - sl.index[-2]) if len(sl) > 1 else pd.Timedelta("1min")
    rb = []
    if is_nse(sym) and not demo: rb = [dict(bounds=["sat", "mon"])] + ([] if tf == "1D" else [dict(bounds=[15.5, 9.25], pattern="hour")])
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000", height=560,
        margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False, dragmode="pan", uirevision=f"{sym}{tf}",
        legend=dict(orientation="h", y=1.04, x=0), yaxis=dict(side="right", gridcolor="#1a1a1a", range=[sl.Low.min() - pad, sl.High.max() + pad]),
        xaxis=dict(gridcolor="#1a1a1a", range=[sl.index[0], sl.index[-1] + step * 8], rangebreaks=rb))
    st.plotly_chart(fig, use_container_width=True, key="main_chart", config={"scrollZoom": True, "displaylogo": False})

@st.fragment(run_every=10)
def orb_block():
    st.markdown('<div class="sec">ORB BASED INDICATION</div>', unsafe_allow_html=True)
    try:
        sym = resolve(st.session_state.q); o = orb_info(sym)
        msg, c = {"ABOVE": ("PRICE CLOSED ABOVE OPENING RANGE – LOOK FOR POSSIBLE LONGS THROUGHOUT THE DAY", G),
                  "BELOW": ("PRICE CLOSED BELOW OPENING RANGE – LOOK FOR POSSIBLE SHORTS THROUGHOUT THE DAY", R),
                  "INSIDE": ("PRICE INSIDE OPENING RANGE – WAIT FOR A 5-MIN CLOSE OUTSIDE THE RANGE", "#ffb300"),
                  "WAIT": ("OPENING RANGE (FIRST 15-MIN CANDLE) NOT YET COMPLETE", "#9aa0a6")}[o["status"]]
        rng = f'<div class="small">Opening range: High {o["hi"]:,.2f} · Low {o["lo"]:,.2f}</div>' if "hi" in o else ""
        mom = "".join(f"<div class='small'>• {m}</div>" for m in momentum(get_ohlc(sym, "1D")[0]))
        st.markdown(f'<div class="panel"><b style="color:{c}">{msg}</b>{rng}<div class="small" style="margin-top:6px;color:#bbb">'
                    f'MOMENTUM NOTE (not a signal)</div>{mom}</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="panel small">ORB data unavailable right now.</div>', unsafe_allow_html=True)

def signal_window():
    st.markdown('<div class="sec">SIGNAL WINDOW</div>', unsafe_allow_html=True)
    sg = st.session_state.get("signals", []); tr = st.session_state.get("trend", "N/A")
    rows = "".join(f'<tr><td>{s["time"]}</td><td>{badge(s["signal"])}'
                   f'{"<div class=small>@ ₹"+format(s["price"],",.2f")+"</div>" if s["price"] else ""}</td>'
                   f'<td>{html.escape(s["reason"])}</td></tr>' for s in sg) or \
           '<tr><td colspan=3 class="small">No confluence at the current price. Signals appear when price reaches a defined level.</td></tr>'
    st.markdown(f'<div class="panel"><div class="small">4H TREND: <b>{tr}</b></div><table class="tbl"><tr><th>TIME</th>'
                f'<th>POSSIBLE SIGNAL</th><th>CONFLUENCE FACTOR USED</th></tr>{rows}</table></div>', unsafe_allow_html=True)

L, Rt = st.columns([3, 2], gap="medium")
with L:
    chart_block()
    _, bc = st.columns([3, 1]); bc.button("GENERATE SIGNALS", on_click=do_gen, use_container_width=True)
with Rt:
    orb_block(); signal_window()

# ---------------- lower sections ----------------
c1, c2, c3 = st.columns([1.4, 1.2, 1.4], gap="medium")
sn = snap(tuple(ASSETS.values())); cr = corrs(sym, tuple(ASSETS.values()))
with c1:
    st.markdown('<div class="sec">CORRELATION MATRIX</div>', unsafe_allow_html=True)
    rows = ""
    for n, t in ASSETS.items():
        p = sn.get(t); cv = cr.get(t)
        cs = f"{cv:+.2f}" if cv is not None and not pd.isna(cv) else "—"
        ps = f'<span style="color:{G if p[1]>=0 else R}">{p[1]:+.2f}%</span>' if p else "—"
        rows += f"<tr><td>{n} {ps}</td><td>{cs}</td><td class='small'>{SECT[n]}</td></tr>"
    st.markdown(f'<div class="panel"><table class="tbl"><tr><th>GLOBAL CATALYST</th><th>CORRELATION (6M)</th>'
                f'<th>AFFECTED SECTORS</th></tr>{rows}</table></div>', unsafe_allow_html=True)
info = fundamentals(sym)
with c2:
    st.markdown('<div class="sec">FUNDAMENTALS</div>', unsafe_allow_html=True)
    d = get_ohlc(sym, "1D")[0].tail(252)
    def nz(k, f=lambda x: f"{x:,.2f}"):
        v = info.get(k); return f(v) if isinstance(v, (int, float)) else "—"
    items = [("P/E ratio", nz("trailingPE")), ("Mkt cap (₹ Cr)", nz("marketCap", lambda x: f"{x/1e7:,.0f}")),
             ("52wk high", nz("fiftyTwoWeekHigh") if info.get("fiftyTwoWeekHigh") else f"{d.High.max():,.2f}"),
             ("EPS", nz("trailingEps")), ("52wk low", nz("fiftyTwoWeekLow") if info.get("fiftyTwoWeekLow") else f"{d.Low.min():,.2f}"),
             ("Revenue (₹ Cr)", nz("totalRevenue", lambda x: f"{x/1e7:,.0f}")), ("Dividend (₹/sh)", nz("dividendRate")),
             ("Revenue growth", nz("revenueGrowth", lambda x: f"{x*100:.1f}%")),
             ("Qtrly earnings growth", nz("earningsQuarterlyGrowth", lambda x: f"{x*100:.1f}%"))]
    st.markdown('<div class="panel"><table class="tbl">' + "".join(
        f"<tr><td class='small'>{k}</td><td><b>{v}</b></td></tr>" for k, v in items) + "</table></div>", unsafe_allow_html=True)
with c3:
    st.markdown('<div class="sec">NEWS FEED</div>', unsafe_allow_html=True)
    nm = info.get("shortName") or sym.replace(".NS", "").replace(".BO", "")
    ns = news(nm)
    body = "".join(f'<div style="margin-bottom:8px"><a href="{html.escape(l or "#")}" target="_blank" style="color:#e6e6e6;'
                   f'text-decoration:none">{i}. {html.escape(t or "")}</a><div class="small">{html.escape(s)} · {html.escape((p or "")[:22])}</div></div>'
                   for i, (t, l, p, s) in enumerate(ns, 1)) or '<div class="small">News unavailable right now.</div>'
    st.markdown(f'<div class="panel" style="max-height:330px;overflow-y:auto">{body}</div>', unsafe_allow_html=True)

d1, d2, d3 = st.columns([1.6, 1, 1], gap="medium")
with d1:
    st.markdown('<div class="sec">INTERNATIONAL MARKETS</div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">' + "".join(tile(n, sn[ASSETS[n]][1] if ASSETS[n] in sn else None)
        for n in ("NIKKEI 225", "GIFT NIFTY*", "NASDAQ", "S&P 500", "DOW JONES", "HANG SENG")) + "</div>"
        '<div class="small">*GIFT Nifty proxied by NIFTY 50 (no free GIFT feed).</div>', unsafe_allow_html=True)
with d2:
    st.markdown('<div class="sec">OTHER ASSETS</div>', unsafe_allow_html=True)
    st.markdown("".join(f'<div style="margin-bottom:8px">{tile(n, sn[ASSETS[n]][1], sn[ASSETS[n]][0]) if ASSETS[n] in sn else tile(n, None)}</div>'
                        for n in ("BRENT CRUDE", "USD/INR")), unsafe_allow_html=True)
with d3:
    st.markdown('<div class="sec">INDIA VIX</div>', unsafe_allow_html=True)
    v = vix()
    if v:
        lab, col = (("RELATIVELY CALM", G) if v[0] < 12 else ("MODERATE", "#ffd54f") if v[0] < 18
                    else ("ELEVATED", "#ff9800") if v[0] < 25 else ("HIGH EXPECTED VOLATILITY", R))
        st.markdown(f'<div class="panel"><span class="ltp">{v[0]:.2f}</span> <b style="color:{col}">{lab}</b>'
                    f'<div class="small">As of {v[1].strftime("%d %b %Y %H:%M IST")}</div>'
                    '<div class="small">Volatility/risk gauge only – not directional; never triggers BUY/SELL.</div></div>', unsafe_allow_html=True)
    else: st.markdown('<div class="panel small">VIX unavailable right now.</div>', unsafe_allow_html=True)
st.caption("Educational analytics tool – not investment advice. Market data via Yahoo Finance; may be delayed.")
