"""Public-safe shared primitives for the conformal-selective-uq substrate.

Self-contained rebuild of a private ancestry project's market-data layer (the
public-safe pieces only), with **its own transparent public universe** — no
account-derived names, conids, or position keys ever enter here (see README).

Lifted, public-safe conventions (unchanged from the ancestry, because PIT-correctness
is the whole game):
  * `known_at` = the bar date's 16:00 ET close, DST-correct via America/New_York,
    stored UTC. No-lookahead is `known_at <= decision_time`, never a hoped-for lag.
  * Trading calendar = XNYS so a missing bar is disambiguated holiday/weekend
    (expected) vs a real coverage hole.
  * yfinance `auto_adjust=False` still returns split-back-adjusted OHLC; we UN-ADJUST
    to as-reported/raw using yfinance's own split column (self-consistent inversion)
    so no future-split lookahead is baked into a level feature, and derive
    split-adjusted RETURNS separately (the future-split factor cancels in a ratio).

NOT run directly — imported by panel.py / synthetic.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # exchange_calendars is the ancestry choice; fall back to a fixed weekday cal
    import exchange_calendars as xc  # type: ignore

    _HAVE_XC = True
except Exception:  # noqa: BLE001
    _HAVE_XC = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROC = DATA / "processed"
for _d in (RAW, INTERIM, PROC):
    _d.mkdir(parents=True, exist_ok=True)

# Persisted artifacts (all gitignored — *.parquet).
PRICES = INTERIM / "public_prices_raw.parquet"  # raw (as-reported) OHLCV + known_at
RETURNS = INTERIM / "public_returns.parquet"  # split-adj daily returns substrate
PANEL = PROC / "public_pit_panel.parquet"  # the PIT feature panel (panel.py output)

# Window: a multi-year public span covering a clear bull leg + a drawdown, so the
# train-pre-drawdown -> test-drawdown shift (G-SHIFT) is real, not synthetic.
WIN_LO = "2018-01-01"
WIN_HI = "2024-12-31"
PULL_END = "2025-01-02"  # yfinance `end` is exclusive

NY = "America/New_York"


# --------------------------------------------------------------------------- #
# Transparent public universe                                                 #
# --------------------------------------------------------------------------- #
# A FIXED, published, public cross-section of liquid US large-caps grouped by
# GICS sector — the demonstration cross-section. Membership is hardcoded (not an
# account-derived list); this carries a known survivorship bias which we state
# honestly (the semi-synthetic Layer-1a allocator is built ON this universe, so
# survivorship does not threaten the *method* claims; 1b forward-returns + Layer-2
# carry external validity). Cap-tier is derived PIT from trailing dollar-volume,
# not a fundamentals feed, so the panel needs no paid data.
SECTORS: dict[str, list[str]] = {
    "tech": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN",
             "AMD", "INTC", "TXN", "QCOM", "IBM"],
    "comm": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"],
    "discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX"],
    "staples": ["WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "MO"],
    "health": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY"],
    "financials": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "C"],
    "industrials": ["CAT", "HON", "UNP", "BA", "GE", "RTX", "DE", "LMT", "UPS", "MMM"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "materials": ["LIN", "APD", "SHW", "FCX", "NEM", "DOW"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "realestate": ["AMT", "PLD", "EQIX", "PSA", "O"],
}

# Public factor / theme proxies (ETFs + macro), exactly the ancestry's spirit:
# broad market, growth, semis, the metals complex, vol, rates. Used for market-model
# betas and as covariate-shift / regime signals. (ticker, role, is_yield)
FACTOR_PROXIES: list[tuple[str, str, bool]] = [
    ("SPY", "broad_market", False),
    ("QQQ", "broad_growth", False),
    ("IWM", "small_cap", False),
    ("MTUM", "momentum_factor", False),
    ("USMV", "lowvol_factor", False),
    ("VLUE", "value_factor", False),
    ("^VIX", "volatility", False),
    ("^TNX", "rates_10y", True),
]
MARKET_TICKER = "SPY"  # market-model market leg


def universe() -> pd.DataFrame:
    """One row per name: ticker, sector. The transparent public cross-section."""
    rows = [{"ticker": t, "sector": sec} for sec, ts in SECTORS.items() for t in ts]
    return pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)


def all_tickers() -> list[str]:
    """Universe names + factor proxies, de-duplicated, for the price pull."""
    names = universe()["ticker"].tolist()
    facs = [t for t, _r, _y in FACTOR_PROXIES]
    return sorted(set(names) | set(facs))


# --------------------------------------------------------------------------- #
# trading calendar + bitemporal stamping                                      #
# --------------------------------------------------------------------------- #
def nyse_sessions(lo: str = WIN_LO, hi: str = WIN_HI) -> pd.DatetimeIndex:
    """Tz-naive XNYS session dates in [lo, hi] (weekday fallback if no xcals)."""
    if _HAVE_XC:
        return xc.get_calendar("XNYS").sessions_in_range(lo, hi).tz_localize(None)
    rng = pd.bdate_range(lo, hi)
    return pd.DatetimeIndex(rng)


def known_at_utc(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Close bars become known at 16:00 ET on the bar date (DST-correct), UTC."""
    d = pd.to_datetime(dates)
    base = d.dt.normalize() if isinstance(d, pd.Series) else d.normalize()
    local = base + pd.Timedelta(hours=16)
    return (
        pd.Series(local)
        .dt.tz_localize(NY, nonexistent="shift_forward", ambiguous=True)
        .dt.tz_convert("UTC")
    )


def yyyymmdd(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%Y%m%d").astype(int)


# --------------------------------------------------------------------------- #
# raw un-adjustment (invert yfinance split-back-adjustment, self-consistent)  #
# --------------------------------------------------------------------------- #
def unadjust_to_raw(h: pd.DataFrame) -> pd.DataFrame:
    """One symbol's yfinance history (auto_adjust=False, actions=True) ->
    as-reported OHLCV + split-adjusted close (for returns).

    raw_level[t] = adj_level[t] * Pi{split_ratio_i : split_date_i > t}.
    The split column is yfinance's own, so the inversion is exact in its frame.
    """
    h = h.sort_index()
    sp = h["Stock Splits"].replace(0.0, np.nan).fillna(1.0)
    rev_cumprod = sp[::-1].cumprod()[::-1]
    cum_future = rev_cumprod / sp  # product of split ratios strictly AFTER t
    out = pd.DataFrame(index=h.index)
    for c in ["Open", "High", "Low", "Close"]:
        out[c.lower()] = h[c] * cum_future
    out["volume"] = (h["Volume"] / cum_future).round()
    out["close_split_adj"] = h["Close"]  # already split-adj by yfinance
    out["stock_split"] = h["Stock Splits"]
    out["dividend"] = h.get("Dividends", 0.0)
    return out


def returns_wide() -> pd.DataFrame:
    """Wide (date x ticker) split-adjusted daily simple returns from RETURNS."""
    r = pd.read_parquet(RETURNS)
    return r.pivot(index="date", columns="ticker", values="ret").sort_index()
