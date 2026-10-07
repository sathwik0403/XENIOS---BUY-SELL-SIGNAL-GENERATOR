import html, time, datetime as dt
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET
import os
import numpy as np, pandas as pd, requests, streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go, yfinance as yf

st.set_page_config(page_title="XENIOS", page_icon="📈", layout="wide")
IST = ZoneInfo("Asia/Kolkata")
G, R = "#00e676", "#ff5252"
OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
_chart = components.declare_component("xenios_chart", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_component"))
TF = {"1m": ("1m", "5d"), "5m": ("5m", "30d"), "15m": ("15m", "60d"),
      "30m": ("30m", "60d"), "4h": ("1h", "730d"), "1D": ("1d", "max")}
ASSETS = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW JONES": "^DJI", "NIKKEI 225": "^N225",
          "HANG SENG": "^HSI", "GIFT NIFTY*": "^NSEI", "BRENT CRUDE": "BZ=F", "USD/INR": "USDINR=X"}
US = ("^GSPC", "^IXIC", "^DJI")
TFMIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}
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
.chg{font-weight:700;font-size:1.1rem}.ltp{font-size:2rem;font-weight:700;line-height:1}.small{color:#9aa0a6;font-size:.75rem}
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
    if "Volume" not in df: df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).fillna(0)
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
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": rng.integers(1000, 9000, n).astype(float)}, index=idx)

def get_ohlc(sym, tf):
    c = st.session_state.get("csv")
    if c is not None:
        if tf == "1D": return c.resample("1D").agg(OHLC).dropna(), False
        if tf == "4h": return resample4h(c), False
        stp = c.index.to_series().diff().median()
        if pd.notna(stp) and stp <= pd.Timedelta(minutes=TFMIN[tf]):
            return c.resample(f"{TFMIN[tf]}min", origin="start_day", offset="9h15min").agg(OHLC).dropna(), False
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
    if "volume" not in d: d["volume"] = 0.0
    d = d[["open", "high", "low", "close", "volume"]].astype(float)
    d.columns = ["Open", "High", "Low", "Close", "Volume"]
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
def nv(t):
    t = pd.Timestamp(t); return t.tz_localize(None) if t.tzinfo else t

def nidx(ix): return ix.tz_localize(None) if getattr(ix, "tz", None) is not None else ix

def fmt_t(t): return nv(t).strftime("%I:%M %p").lstrip("0")

def atr(df, n=14):
    if len(df) < 2: return 1.0
    pc = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(n, min_periods=2).mean().iloc[-1]
    return float(v) if pd.notna(v) and v > 0 else float((df.High - df.Low).mean() or 1.0)

def pivots(df, k=2):
    """Fractal swing pivots, alternated H/L, tiny swings (<0.5 ATR) ignored."""
    h, l, o, c = df.High.values, df.Low.values, df.Open.values, df.Close.values; P = []
    for i in range(k, len(df) - k):
        isH = h[i] > h[i - k:i].max() and h[i] >= h[i + 1:i + k + 1].max()
        isL = l[i] < l[i - k:i].min() and l[i] <= l[i + 1:i + k + 1].min()
        pr = ([(i, l[i], "L")] if isL else []) + ([(i, h[i], "H")] if isH else [])
        if isL and isH and c[i] < o[i]: pr = pr[::-1]
        P += pr
    a = atr(df); out = []
    for p in P:
        if out and out[-1][2] == p[2]:
            if (p[2] == "H" and p[1] > out[-1][1]) or (p[2] == "L" and p[1] < out[-1][1]): out[-1] = p
        elif out and abs(p[1] - out[-1][1]) < .5 * a: continue
        else: out.append(p)
    return out

def structure(h4):
    """Label 4H pivots HH/LH/HL/LL, derive trend and the Fibonacci leg."""
    h = h4.tail(400); piv = pivots(h); lab = []; lh = ll = None
    for i, pr, t in piv:
        if t == "H": lb = None if lh is None else ("HH" if pr > lh else "LH"); lh = pr
        else: lb = None if ll is None else ("HL" if pr > ll else "LL"); ll = pr
        lab.append((i, pr, t, lb))
    Hs = [x for x in lab if x[2] == "H" and x[3]]; Ls = [x for x in lab if x[2] == "L" and x[3]]
    trend, leg, ix = "SIDEWAYS", None, h.index
    if Hs and Ls:
        if Hs[-1][3] == "HH" and Ls[-1][3] == "HL": trend = "UPTREND"
        elif Hs[-1][3] == "LH" and Ls[-1][3] == "LL": trend = "DOWNTREND"
    if trend == "UPTREND":
        hh = Hs[-1]; lows = [x for x in lab if x[2] == "L" and x[0] < hh[0]]
        hl = ([x for x in lows if x[3] == "HL"] or lows or [None])[-1]
        if hl: leg = dict(lo=hl[1], hi=hh[1], dr="up", t0=nv(ix[hl[0]]), t1=nv(ix[hh[0]]))
    elif trend == "DOWNTREND":
        lw = Ls[-1]; highs = [x for x in lab if x[2] == "H" and x[0] < lw[0]]
        lhh = ([x for x in highs if x[3] == "LH"] or highs or [None])[-1]
        if lhh: leg = dict(lo=lw[1], hi=lhh[1], dr="down", t0=nv(ix[lhh[0]]), t1=nv(ix[lw[0]]))
    return dict(trend=trend, leg=leg, lab=[(nv(ix[i]), pr, t, lb) for i, pr, t, lb in lab], piv=piv, h=h)

FIB_UP = {0.382: "NORMAL BUY", 0.5: "MODERATE BUY", 0.618: "STRONG BUY", 0.786: "MODERATE SELL"}
FIB_DN = {0.382: "NORMAL SELL", 0.5: "MODERATE SELL", 0.618: "STRONG SELL", 0.786: "MODERATE BUY"}
FIB_ATH = {0.382: "SUFFICIENT BUY", 0.5: "MODERATE BUY", 0.618: "STRONG BUY"}

def fib_price(leg, p):
    return leg["hi"] - p * (leg["hi"] - leg["lo"]) if leg["dr"] == "up" else leg["lo"] + p * (leg["hi"] - leg["lo"])

def psy_step(p): return 1 if p < 20 else 5 if p < 100 else 50 if p < 1000 else 100 if p < 10000 else 500
def strength(a): return "WEAK" if a < .6 else "NORMAL" if a < 1 else "MODERATE" if a < 1.5 else "GOOD" if a < 2.5 else "STRONG"

def last_bos(h, piv):
    C = h.Close.values; best = None
    for i, pr, t in piv:
        for j in range(i + 1, len(C)):
            if (t == "H" and C[j] > pr) or (t == "L" and C[j] < pr):
                if best is None or j > best[0]: best = (j, "bull" if t == "H" else "bear")
                break
    return best

def fvgs(h, a):
    H, L, C = h.High.values, h.Low.values, h.Close.values; o = []
    for i in range(len(h) - 2):
        if L[i + 2] > H[i] and L[i + 2] - H[i] >= .1 * a: o.append(dict(i=i + 1, dir="bull", lo=H[i], hi=L[i + 2]))
        elif H[i + 2] < L[i] and L[i] - H[i + 2] >= .1 * a: o.append(dict(i=i + 1, dir="bear", lo=H[i + 2], hi=L[i]))
    for g in o:
        g["inv"] = None
        for j in range(g["i"] + 2, len(h)):
            if (g["dir"] == "bull" and C[j] < g["lo"]) or (g["dir"] == "bear" and C[j] > g["hi"]): g["inv"] = j; break
    return o

def order_blocks(h, a):
    O, C, H, L = h.Open.values, h.Close.values, h.High.values, h.Low.values; obs = []
    for i in range(1, len(h) - 3):
        if C[i] < O[i] and C[i + 1] > O[i + 1] and C[i + 1:i + 4].max() > H[i] and H[i + 1:i + 4].max() - L[i] > 1.5 * a:
            obs.append(dict(dir="bull", i=i, lo=L[i], hi=H[i]))
        if C[i] > O[i] and C[i + 1] < O[i + 1] and C[i + 1:i + 4].min() < L[i] and H[i] - L[i + 1:i + 4].min() > 1.5 * a:
            obs.append(dict(dir="bear", i=i, lo=L[i], hi=H[i]))
    for ob in obs:
        ob["fresh"] = True; ob["t"] = nv(h.index[ob["i"]])
        for j in range(ob["i"] + 4, len(h) - 1):
            if L[j] <= ob["hi"] and H[j] >= ob["lo"]: ob["fresh"] = False; break
    return obs

def vol_profile(day, rows=240):
    lo, hi = float(day.Low.min()), float(day.High.max())
    if hi <= lo: return None
    edges = np.linspace(lo, hi, rows + 1); vol = np.zeros(rows); has = day.Volume.sum() > 0
    for h_, l_, v in zip(day.High.values, day.Low.values, day.Volume.values if has else np.ones(len(day))):
        i0 = max(np.searchsorted(edges, l_, side="right") - 1, 0); i1 = min(np.searchsorted(edges, h_, side="right") - 1, rows - 1)
        vol[i0:i1 + 1] += v / (i1 - i0 + 1)
    poc = int(vol.argmax()); acc = vol[poc]; a_, b_ = poc, poc; tot = vol.sum()
    while acc < .7 * tot:
        up = vol[b_ + 1] if b_ + 1 < rows else -1; dn = vol[a_ - 1] if a_ > 0 else -1
        if up < 0 and dn < 0: break
        if up >= dn: b_ += 1; acc += up
        else: a_ -= 1; acc += dn
    pf = (edges[poc] + edges[poc + 1]) / 2
    fr = (pf - lo) / (hi - lo); sm = np.convolve(vol, np.ones(5) / 5, mode="same")
    return dict(edges=edges, vol=vol, sm=sm, mean=float(sm.mean()), poc=float(pf), vah=float(edges[b_ + 1]), val=float(edges[a_]),
                has=bool(has), shape="P" if fr > .65 else "b" if fr < .35 else "D")

def vp_class(vp, price):
    e = vp["edges"]
    if price < e[0] or price > e[-1]: return "OUT"
    i = int(np.clip(np.searchsorted(e, price, side="right") - 1, 0, len(vp["sm"]) - 1)); v = vp["sm"][i]
    return "LVN" if v < .3 * vp["mean"] else "HVN" if v > 1.25 * vp["mean"] else "MID"

def ema_bias(h):
    c = h.Close; e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1]); e200 = float(c.ewm(span=200, adjust=False).mean().iloc[-1]); p = float(c.iloc[-1])
    return ("bull" if e50 > e200 and p > e200 else "bear" if e50 < e200 and p < e200 else "mixed"), e50, e200, p

def analyze(sym):
    out = dict(sigs=[], trend="N/A", lab=[], leg=None, obs=[], bf=None, vp=None, report="No data available.")
    h4 = get_ohlc(sym, "4h")[0]; m5 = get_ohlc(sym, "5m")[0]; d = get_ohlc(sym, "1D")[0]; m1 = get_ohlc(sym, "1m")[0]
    ltp = float(m1.Close.iloc[-1]); a4 = atr(h4); tnow = nv(m1.index[-1])
    S = structure(h4); trend, leg = S["trend"], S["leg"]; out.update(trend=trend, leg=leg, lab=S["lab"])
    bias, e50, e200, c4 = ema_bias(h4)
    ses = {dd: g for dd, g in m5.groupby(m5.index.date)}; days = sorted(ses)
    cur = ses[days[-1]] if days else m5.iloc[0:0]; prev = ses[days[-2]] if len(days) > 1 else None
    pdh = pdl = pdc = None
    if prev is not None and len(prev): pdh, pdl, pdc = float(prev.High.max()), float(prev.Low.min()), float(prev.Close.iloc[-1])
    vp = vol_profile(prev) if prev is not None and len(prev) > 3 else None
    if vp: vp["t0"] = nv(prev.index[0]); out["vp"] = vp
    ht = h4.tail(150); obs = order_blocks(ht, a4); fresh = [o for o in obs if o["fresh"]]; out["obs"] = fresh[-4:]
    fv = fvgs(ht, a4)
    ifv = [dict(pol="bull" if g["dir"] == "bear" else "bear", lo=g["lo"], hi=g["hi"]) for g in fv if g["inv"] is not None]
    bos = last_bos(ht, pivots(ht)); bf = None
    if bos:
        c_ = [g for g in fv if g["dir"] == bos[0 + 1] and bos[0] - 1 <= g["i"] <= bos[0] + 2]
        if c_:
            g = c_[0]; b_bar = ht.iloc[bos[0]]
            bf = dict(g, bd=bos[1], t=nv(ht.index[g["i"]]), valid=g["inv"] is None,
                      sl_ref=float(b_bar.Low if bos[1] == "bull" else b_bar.High))
    out["bf"] = bf
    # --- AMD / liquidity sweeps on 5m (accumulation = first 30 min of the session)
    sweeps = []; accH = accL = None
    if len(cur) > 6:
        accH, accL = float(cur.High.iloc[:6].max()), float(cur.Low.iloc[:6].min())
        hi_l = [("Accumulation High", accH)] + ([("PDH", pdh)] if pdh else []); lo_l = [("Accumulation Low", accL)] + ([("PDL", pdl)] if pdl else [])
        used = set()
        for t, b in cur.iloc[6:].iterrows():
            for nm, v in hi_l:
                if nm not in used and b.High > v and b.Close < v: sweeps.append(dict(t=nv(t), dir="bear", name=nm, lvl=v, bar=b)); used.add(nm)
            for nm, v in lo_l:
                if nm not in used and b.Low < v and b.Close > v: sweeps.append(dict(t=nv(t), dir="bull", name=nm, lvl=v, bar=b)); used.add(nm)
    sigs = []
    def add(t, sig, price, why, valid, sl=None, kind="", p=None):
        sigs.append(dict(t=nv(t), signal=sig, price=price, reason=why, valid=valid, sl=sl, kind=kind, p=p))
    def filt(want, price, t, rev, p=None):
        ok, bad = [], []; tol = .1 * a4
        sw = [s for s in sweeps if s["t"] <= nv(t) and s["dir"] == want]
        if sw: ok.append(f"{'SSL' if want == 'bull' else 'BSL'} sweep {fmt_t(sw[-1]['t'])}")
        for ob in fresh:
            if ob["lo"] - tol <= price <= ob["hi"] + tol: (ok if ob["dir"] == want else bad).append(f"fresh 4H {ob['dir']} OB")
        for z in ifv:
            if z["lo"] - tol <= price <= z["hi"] + tol: (ok if z["pol"] == want else bad).append("iFVG flipped zone")
            elif not rev and ((want == "bull" and z["pol"] == "bear" and 0 < z["lo"] - price < a4) or (want == "bear" and z["pol"] == "bull" and 0 < price - z["hi"] < a4)):
                bad.append("opposing iFVG in path")
        if not rev and bias in ("bull", "bear"): (ok if bias == want else bad).append(f"4H EMA50/200 {bias}")
        if vp:
            cl_ = vp_class(vp, price)
            if cl_ == "LVN": bad.append("inside Low-Volume Node")
            elif p == .382 and cl_ != "HVN": bad.append("38.2% not on HVN shelf")
            elif cl_ == "HVN": ok.append("on HVN shelf")
        return not bad, list(dict.fromkeys(ok)), list(dict.fromkeys(bad))
    # --- level book
    levels = []
    def LV(key, price, tol, sig, why, kind, p=None, rev=False, sl=None): levels.append(dict(key=key, price=float(price), tol=tol, sig=sig, why=why, kind=kind, p=p, rev=rev, sl=sl))
    broken = bool(leg) and ((leg["dr"] == "up" and ltp < leg["lo"]) or (leg["dr"] == "down" and ltp > leg["hi"]))
    if leg and not broken:
        for p, sg in (FIB_UP if leg["dr"] == "up" else FIB_DN).items():
            LV(f"fib{p}", fib_price(leg, p), ltp * .001, sg, f"{p * 100:.1f}% Fib retracement on 4H {trend.lower()}", "fib", p, p == .786,
               (leg["lo"] - .1 * a4) if "BUY" in sg else (leg["hi"] + .1 * a4))
    if trend == "DOWNTREND":
        dc = d.Close.iloc[:-1] if len(d) > 1 else d.Close
        for n, sg in ((21, "NORMAL SELL"), (50, "GOOD SELL"), (200, "STRONG SELL")):
            v = dc.rolling(n).mean().iloc[-1]
            if pd.notna(v): LV(f"sma{n}", v, ltp * .002, sg, f"Price near daily {n} SMA in downtrend – bearish bounce likely", "sma")
        ath, atl = float(d.High.max()), float(d.Low.min())
        for p, sg in FIB_ATH.items(): LV(f"ath{p}", ath - p * (ath - atl), ltp * .002, sg, f"{p * 100:.1f}% Fib level from ATH {ath:,.2f} (downtrend)", "ath")
    if bf and bf["valid"]:
        bu = bf["bd"] == "bull"; mid_ = (bf["lo"] + bf["hi"]) / 2
        LV("fvg", mid_, (bf["hi"] - bf["lo"]) / 2 + ltp * .0005, "MODERATE BUY" if bu else "MODERATE SELL",
           "First FVG after 4H break of structure (breakout FVG retest)", "fvg", sl=bf["sl_ref"] - .1 * a4 if bu else bf["sl_ref"] + .1 * a4)
    def emit(lv, t):
        bull = "BUY" in lv["sig"]
        if lv["kind"] == "fib":
            ok_, ok, bad = filt("bull" if bull else "bear", lv["price"], t, lv["rev"], lv["p"])
            why = lv["why"] + " | " + " · ".join(["✔ " + x for x in ok] + ["✖ " + x for x in bad])
            add(t, lv["sig"], lv["price"], why.rstrip(" |"), ok_, lv["sl"], "fib", lv["p"])
        else:
            okk = lv["sl"] is None or (lv["sl"] < lv["price"] if bull else lv["sl"] > lv["price"])
            add(t, lv["sig"], lv["price"], lv["why"] + ("" if okk else " | ✖ price already beyond breakout candle"), okk, lv["sl"], lv["kind"])
    rng5 = float((m5.High - m5.Low).tail(375).mean()) if len(m5) else ltp * .002
    vic = min(max(1.5 * rng5, ltp * .001), ltp * .005); step = psy_step(ltp); state = {}
    for t, b in cur.iterrows():
        for lv in levels:
            hit = b.Low - lv["tol"] <= lv["price"] <= b.High + lv["tol"]; k_ = lv["key"]
            if k_ not in state: state[k_] = True
            if hit and state[k_]: emit(lv, t); state[k_] = False
            elif abs(float(b.Close) - lv["price"]) > 4 * lv["tol"]: state[k_] = True
        if trend in ("UPTREND", "DOWNTREND"):
            cl = float(b.Close)
            lvl = np.ceil(cl / step) * step if trend == "UPTREND" else np.floor(cl / step) * step
            near = (0 <= lvl - cl <= vic) if trend == "UPTREND" else (0 <= cl - lvl <= vic); k = ("psy", lvl)
            if near and not state.get(k):
                add(t, "STRONG BUY" if trend == "UPTREND" else "STRONG SELL", float(lvl),
                    f"Price {cl:,.2f} {'approaching' if trend == 'UPTREND' else 'falling to'} psychological level {lvl:,.0f} ({trend.lower()} magnet)", True, kind="psy")
            state[k] = near
    if len(cur) and pdc:
        o0 = float(cur.Open.iloc[0]); gp = (o0 / pdc - 1) * 100
        if .25 <= abs(gp) <= .5:
            bu = gp < 0; dist = abs(o0 - pdc)
            add(cur.index[0], "NORMAL BUY" if bu else "NORMAL SELL", o0, f"Opening gap {gp:+.2f}% – gap-fade toward yesterday's close {pdc:,.2f} (SL {dist:,.2f} beyond open)",
                True, o0 - dist if bu else o0 + dist, "gap")
    for s in sweeps:
        w = s["dir"]
        if bias == ("bear" if w == "bull" else "bull"): continue
        obh = any(ob["dir"] == w and ob["lo"] - .1 * a4 <= s["bar"].Close <= ob["hi"] + .1 * a4 for ob in fresh)
        st_ = "STRONG" if obh else "MODERATE" if bias == w else "NORMAL"
        add(s["t"], f"{st_} {'BUY' if w == 'bull' else 'SELL'}", float(s["bar"].Close),
            f"{'SSL' if w == 'bull' else 'BSL'} sweep of {s['name']} {s['lvl']:,.2f} (AMD manipulation)" + (" + reaction from fresh 4H OB" if obh else "") + (f" + 4H EMA {bias}" if bias == w else ""),
            True, float(s["bar"].Low * .9995 if w == "bull" else s["bar"].High * 1.0005), "amd")
    try:
        sn = snap(tuple(ASSETS.values())); cr = corrs(sym, tuple(ASSETS.values()))
        for name, t_ in ASSETS.items():
            if t_ not in sn or t_ not in cr or t_ == sym or pd.isna(cr[t_]) or abs(cr[t_]) < .25: continue
            pct = sn[t_][1]; a_ = abs(pct) * (2.5 if name == "USD/INR" else 1)
            if a_ < .3: continue
            buy = np.sign(pct) * np.sign(cr[t_]) > 0
            add(tnow, f"{strength(a_)} {'BUY' if buy else 'SELL'}", None, f"{name} {pct:+.2f}% · 6M correlation {cr[t_]:+.2f} with stock", True, kind="macro")
    except Exception: pass
    sigs.sort(key=lambda s: s["t"], reverse=True); out["sigs"] = sigs[:40]
    # --- structural analysis matrix (report)
    tgt = next((s for s in sigs if s["kind"] == "fib" and s["valid"]), None) or next((s for s in sigs if s["valid"] and s["sl"] is not None), None) \
        or next((s for s in sigs if s["kind"] == "fib"), None)
    bt = {"bull": "Bullish", "bear": "Bearish", "mixed": "Mixed"}[bias]
    ema_t = f"{'Above' if c4 > max(e50, e200) else 'Below' if c4 < min(e50, e200) else 'Between'} 50/200 EMA ({e50:,.2f} / {e200:,.2f}) – {bt} alignment"
    if sweeps:
        s0 = sweeps[-1]; amd_t = (f"Accumulation {accL:,.2f}–{accH:,.2f} → Manipulation: {'SSL' if s0['dir'] == 'bull' else 'BSL'} swept {s0['name']} {s0['lvl']:,.2f} at {fmt_t(s0['t'])} → "
                                  f"Distribution {'up' if s0['dir'] == 'bull' else 'down'}{' underway' if (ltp > s0['bar'].Close) == (s0['dir'] == 'bull') else ' stalled'}")
    elif accH: amd_t = f"Accumulation phase – range {accL:,.2f}–{accH:,.2f}; no liquidity sweep yet"
    else: amd_t = "Session too young / intraday data unavailable"
    near_ob = next((o for o in reversed(fresh) if o["lo"] - .1 * a4 <= ltp <= o["hi"] + .1 * a4), None)
    ob_t = (f"Reacting at fresh 4H {'Bullish' if near_ob['dir'] == 'bull' else 'Bearish'} Order Block {near_ob['lo']:,.2f}–{near_ob['hi']:,.2f}" if near_ob
            else (f"No fresh 4H OB at price; nearest: {min(fresh, key=lambda o: abs((o['lo'] + o['hi']) / 2 - ltp))['dir']} OB" if fresh else "No fresh 4H Order Block"))
    ref = tgt["price"] if tgt and tgt["price"] else ltp
    vp_t = (f"{vp['shape']}-Shape Profile{'' if vp['has'] else ' (TPO – volume unavailable)'} | POC: {vp['poc']:,.2f} | VAH: {vp['vah']:,.2f} | VAL: {vp['val']:,.2f} | Level is on a {vp_class(vp, ref)} zone"
            if vp else "Previous-day profile unavailable")
    if bf: fv_t = ("First FVG post-breakout confirmed" if bf["valid"] else "Breakout FVG already inverted") + f" ({bf['lo']:,.2f}–{bf['hi']:,.2f})"
    else: fv_t = "No breakout FVG sequence"
    if any(g["inv"] is None and g["lo"] <= ltp <= g["hi"] and not (bf and g["i"] == bf.get("i")) for g in fv): fv_t += " | Non-compliant mid-range FVG detected → INVALIDATED"
    if leg:
        coords = (f"HH: {leg['hi']:,.2f} | HL: {leg['lo']:,.2f}" if leg["dr"] == "up" else f"LH: {leg['hi']:,.2f} | LL: {leg['lo']:,.2f}")
    else: coords = "No valid HH/HL or LH/LL sequence (sideways)"
    if tgt and tgt["kind"] == "fib": tp = f"{tgt['p'] * 100:.1f}% Retracement Level touched at {tgt['price']:,.2f} ({fmt_t(tgt['t'])})"
    else: tp = f"No Fibonacci retracement level touched (price {ltp:,.2f})"
    if tgt: act = tgt["signal"] if tgt["valid"] else "INVALIDATED - NO ENTRY"; ent = f"{tgt['price']:,.2f}" if tgt["price"] else "—"; slv = f"{tgt['sl']:,.2f}" if tgt["sl"] else "—"
    else: act, ent, slv = "INVALIDATED - NO ENTRY", "—", "—"
    sl_why = "beyond the 4H structural swing wick (leg extreme ∓ 0.1×ATR)" if tgt and tgt["kind"] == "fib" else "per signal rule"
    out["report"] = (f"**{sym.replace('.NS', '')} – STRUCTURAL ANALYSIS MATRIX**\n\n**1. HIERARCHY LAYER VERIFICATION**\n"
        f"- **EMA Macro Bias Status:** {ema_t}\n- **AMD Phase & Sweeps:** {amd_t}\n- **Order Block & Flow Check:** {ob_t}\n"
        f"- **Volume Profile Architecture:** {vp_t}\n- **Breakout FVG Sequence:** {fv_t}\n\n**2. 4H TREND & FIBONACCI STATE**\n"
        f"- **Marked 4H Trend Coordinates:** [{coords}]  ({trend})\n- **Current Retracement Touchpoint:** {tp}\n\n**3. FINAL TRADING VERDICT**\n"
        f"- **Signal Action:** {act}\n- **Optimal Entry Target:** {ent}\n- **Mathematical Invalidation (Stop Loss):** {slv}" + (f" – {sl_why}" if slv != "—" else ""))
    return out

def get_analysis(sym):
    c = st.session_state.get("csv"); cs = None if c is None else (len(c), str(c.index[-1]))
    key = (sym, int(time.time() // 10), cs, st.session_state.get("gen_n", 0)); m = st.session_state.get("_an")
    if m and m[0] == key: return m[1]
    try: res = analyze(sym)
    except Exception as e:
        res = dict(sigs=[], trend="N/A", lab=[], leg=None, obs=[], bf=None, vp=None, report=f"Analysis unavailable ({type(e).__name__}).")
    st.session_state._an = (key, res); return res

def do_gen(): st.session_state.gen_n = st.session_state.get("gen_n", 0) + 1

def orb_info(sym):
    m, _ = get_ohlc(sym, "5m"); day = m.index[-1].date()
    s = m[[i.date() == day for i in m.index]]
    if pd.Timestamp.now(tz=s.index.tz) - s.index[-1] < pd.Timedelta("5min"): s = s.iloc[:-1]
    if len(s) < 3: return {"status": "WAIT"}
    hi, lo = s.High.iloc[:3].max(), s.Low.iloc[:3].min(); st_ = "INSIDE"; brk = None
    for t, c in s.Close.iloc[3:6].items():          # 5-min closes inside first 30 min since open
        if c > hi: st_, brk = "ABOVE", t; break
        if c < lo: st_, brk = "BELOW", t; break
    if st_ == "INSIDE" and len(s) >= 6: st_ = "NONE"
    return dict(status=st_, hi=hi, lo=lo, start=s.index[0], brk=brk)

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
    chg = ltp - prev; pct = (ltp / prev - 1) * 100; c = G if chg >= 0 else R
    tag = '<span style="color:#ffb300">● DEMO DATA (live feed unavailable)</span>' if demo else \
          f'<span style="color:{G}">● LIVE</span> · last tick {df.index[-1].strftime("%d %b %H:%M")}'
    st.markdown(f'<div class="small">LTP</div><span class="ltp">{ltp:,.2f}</span> '
                f'<span class="chg" style="color:{c}">{"▲" if chg >= 0 else "▼"} {chg:+.2f}</span> '
                f'<span class="chg" style="color:{c}">({pct:+.2f}%)</span>'
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
r1, r2, r3 = st.columns([1.1, 2.6, 3])
r1.markdown(f'<div class="sec" style="font-size:1.1rem;color:#fff">{sym}</div>', unsafe_allow_html=True)
r2.radio("tf", list(TF), index=1, horizontal=True, key="tf", label_visibility="collapsed")
with r3:
    k1, k2, k3, k4 = st.columns(4)
    k1.checkbox("SMA", True, key="t_sma"); k2.checkbox("FIB/HH-LL", True, key="t_fib"); k3.checkbox("ORB", True, key="t_orb"); k4.checkbox("VP/SMC", True, key="t_vp")

# ---------------- chart ----------------
@st.fragment(run_every=3)
def chart_block():
    sym = resolve(st.session_state.q); tf = st.session_state.tf
    df, demo = get_ohlc(sym, tf); d = get_ohlc(sym, "1D")[0]; an = get_analysis(sym)
    x = nidx(df.index); xe = x[-1]
    step = (x[-1] - x[-2]) if len(x) > 1 else pd.Timedelta("1min"); xr = xe + step * 8
    fig = go.Figure(go.Candlestick(x=x, open=df.Open, high=df.High, low=df.Low, close=df.Close, name=sym,
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"), showlegend=False))
    if st.session_state.t_sma:
        for n, col in ((21, "#f5c518"), (50, "#2196f3"), (200, "#e040fb")):
            fig.add_trace(go.Scatter(x=x, y=sma_line(df, d, tf, n).values, name=f"SMA {n}", mode="lines", line=dict(color=col, width=1.4)))
    if tf == "4h" and st.session_state.t_fib:
        pts = [(t, p, ty, lb) for t, p, ty, lb in an["lab"] if lb and t >= x[0]]
        if pts:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers+text", showlegend=False, hoverinfo="skip",
                text=[p[3] for p in pts], textposition=["top center" if p[2] == "H" else "bottom center" for p in pts],
                textfont=dict(size=11, color=[G if p[3] in ("HH", "HL") else R for p in pts]),
                marker=dict(size=6, color=[G if p[3] in ("HH", "HL") else R for p in pts])))
        leg = an["leg"]
        if leg:
            up = leg["dr"] == "up"
            fig.add_shape(type="line", x0=leg["t0"], y0=leg["lo"] if up else leg["hi"], x1=leg["t1"], y1=leg["hi"] if up else leg["lo"],
                          line=dict(color="#ffd54f", width=1.2, dash="dash"))
            for p in (0, .236, .382, .5, .618, .786, 1):
                y = fib_price(leg, p)
                fig.add_shape(type="line", x0=leg["t1"], x1=xr, y0=y, y1=y, line=dict(color="#888", width=.8, dash="dot"))
                fig.add_annotation(x=xr, y=y, text=f"{p * 100:.1f}%  {y:,.2f}", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="#aaa", size=9))
    if tf == "4h" and st.session_state.t_vp:
        for ob in an["obs"]:
            c = "0,230,118" if ob["dir"] == "bull" else "255,82,82"
            fig.add_shape(type="rect", x0=ob["t"], x1=xr, y0=ob["lo"], y1=ob["hi"], fillcolor=f"rgba({c},.15)", line=dict(width=0))
            fig.add_annotation(x=xr, y=ob["hi"], text="OB", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=f"rgb({c})", size=9))
        bf = an["bf"]
        if bf and bf["valid"]:
            fig.add_shape(type="rect", x0=bf["t"], x1=xr, y0=bf["lo"], y1=bf["hi"], fillcolor="rgba(255,213,79,.15)", line=dict(width=0))
            fig.add_annotation(x=xr, y=bf["hi"], text="FVG", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="#ffd54f", size=9))
    vp = an["vp"]
    if vp and tf in ("1m", "5m", "15m", "30m") and st.session_state.t_vp:
        s0 = vp["t0"]; w = pd.Timedelta(hours=2); e = vp["edges"]; mx = vp["vol"].max() or 1
        xs, ys = [s0], [e[0]]
        for i, v in enumerate(vp["vol"]):
            xs += [s0 + w * (v / mx)] * 2; ys += [e[i], e[i + 1]]
        xs.append(s0); ys.append(e[-1])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor="rgba(66,165,245,.30)", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        for y, t, col in ((vp["poc"], "POC", "#ff9800"), (vp["vah"], "VAH", "#90caf9"), (vp["val"], "VAL", "#90caf9")):
            fig.add_shape(type="line", x0=s0, x1=xr, y0=y, y1=y, line=dict(color=col, width=1, dash="solid" if t == "POC" else "dot"))
            fig.add_annotation(x=xr, y=y, text=f"PD {t} {y:,.2f}", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=col, size=9))
    if st.session_state.t_orb and tf in ("1m", "5m", "15m", "30m"):
        try:
            o = orb_info(sym)
            if o["status"] != "WAIT":
                for y, t in ((o["hi"], "OR HIGH"), (o["lo"], "OR LOW")):
                    fig.add_shape(type="line", x0=nv(o["start"]), x1=xe, y0=y, y1=y, line=dict(color="#ffb300", width=1.2, dash="dash"))
                    fig.add_annotation(x=xe, y=y, text=t, showarrow=False, font=dict(color="#ffb300", size=10), xanchor="left")
        except Exception: pass
    for s in [s for s in an["sigs"] if s["valid"] and s["price"]][:5]:
        c = G if "BUY" in s["signal"] else R
        fig.add_hline(y=s["price"], line_dash="dot", line_width=1, line_color=c, annotation_text=s["signal"], annotation_font_color=c, annotation_position="top left")
    vis = {"1m": 150, "5m": 150, "15m": 140, "30m": 120, "4h": 120, "1D": 150}[tf]; sl = df.tail(vis)
    pad = (sl.High.max() - sl.Low.min()) * .08 or 1
    rb = []
    if is_nse(sym) and not demo: rb = [dict(bounds=["sat", "mon"])] + ([] if tf == "1D" else [dict(bounds=[15.5, 9.25], pattern="hour")])
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000", margin=dict(l=5, r=5, t=10, b=5),
        xaxis_rangeslider_visible=False, dragmode="pan", uirevision=f"{sym}{tf}{st.session_state.get('csv') is not None}",
        legend=dict(orientation="h", y=1.04, x=0),
        yaxis=dict(side="right", gridcolor="#1a1a1a", range=[sl.Low.min() - pad, sl.High.max() + pad], fixedrange=False),
        xaxis=dict(gridcolor="#1a1a1a", range=[x[-min(vis, len(x))], xr], rangebreaks=rb))
    _chart(fig_json=fig.to_json(), height=560, key="main_chart", default=None)

@st.fragment(run_every=10)
def orb_block():
    st.markdown('<div class="sec">ORB BASED INDICATION</div>', unsafe_allow_html=True)
    try:
        sym = resolve(st.session_state.q); o = orb_info(sym)
        msg, c = {"ABOVE": ("PRICE CLOSED ABOVE OPENING RANGE IN FIRST 30 MIN – LOOK FOR POSSIBLE LONGS THROUGHOUT THE DAY", G),
                  "BELOW": ("PRICE CLOSED BELOW OPENING RANGE IN FIRST 30 MIN – LOOK FOR POSSIBLE SHORTS THROUGHOUT THE DAY", R),
                  "INSIDE": ("PRICE INSIDE OPENING RANGE – WAITING FOR A 5-MIN CLOSE OUTSIDE IT (FIRST 30 MIN WINDOW)", "#ffb300"),
                  "NONE": ("NO CLOSE OUTSIDE OPENING RANGE IN FIRST 30 MIN – NO ORB BIAS TODAY", "#9aa0a6"),
                  "WAIT": ("OPENING RANGE (FIRST 15-MIN CANDLE) NOT YET COMPLETE", "#9aa0a6")}[o["status"]]
        rng = f'<div class="small">Opening range: High {o["hi"]:,.2f} · Low {o["lo"]:,.2f}' + (f' · breakout close {o["brk"].strftime("%H:%M")}' if o.get("brk") is not None else "") + "</div>" if "hi" in o else ""
        mom = "".join(f"<div class='small'>• {m}</div>" for m in momentum(get_ohlc(sym, "1D")[0]))
        st.markdown(f'<div class="panel"><b style="color:{c}">{msg}</b>{rng}<div class="small" style="margin-top:6px;color:#bbb">'
                    f'MOMENTUM NOTE (not a signal)</div>{mom}</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="panel small">ORB data unavailable right now.</div>', unsafe_allow_html=True)

@st.fragment(run_every=10)
def signal_window():
    st.markdown('<div class="sec">SIGNAL WINDOW</div>', unsafe_allow_html=True)
    an = get_analysis(resolve(st.session_state.q)); sg = an["sigs"]
    def row(s):
        sig = badge(s["signal"]) if s["valid"] else f'<span class="badge" style="color:#9aa0a6;border-color:#555">INVALIDATED – NO ENTRY</span><div class="small">({html.escape(s["signal"])})</div>'
        px = f'<div class="small">@ ₹{s["price"]:,.2f}</div>' if s["price"] else ""
        return (f'<tr><td>{fmt_t(s["t"])}<div class="small">{s["t"].strftime("%d %b")}</div></td><td>{sig}{px}</td><td>{html.escape(s["reason"])}</td></tr>')
    rows = "".join(row(s) for s in sg) or '<tr><td colspan=3 class="small">No confluence found in the latest session.</td></tr>'
    st.markdown(f'<div class="panel" style="max-height:420px;overflow-y:auto"><div class="small">4H TREND: <b>{an["trend"]}</b> · times = candle time in the trading session</div>'
                f'<table class="tbl"><tr><th>TIME</th><th>POSSIBLE SIGNAL</th><th>CONFLUENCE FACTOR USED</th></tr>{rows}</table></div>', unsafe_allow_html=True)
    with st.expander("SMC STRUCTURAL ANALYSIS MATRIX"): st.markdown(an["report"])

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
