"""polymarket-nvidia-ceo-forecast-01 entry: centred empirical quantile forecaster.

For every market of Polymarket's "Next CEO of NVIDIA?" event, forecast the Yes
price for the next 240 minutes as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

Method (no model call, no network, stdlib only) — final, chosen by backtest
grid over 6 practice cases (own scorer, validated: it reproduces the house
persistence baseline practice loss 0.363589 exactly):
  - MEDIAN: mean-reversion toward the market's fair value (its last price
    divided by the sum of all markets' last prices — the 24 Yes prices sum
    to >1, the overround), with a ~2h half-life: fair + (x - fair)*exp(-h/120).
    The long-run (72h) mean is a TERRIBLE target (tested: 0.955) — the price's
    recent level is what matters.
  - SPREAD: the empirical h-minute changes over the RECENT 12h, CENTRED by
    their own median (so the median stays the mean-reverting one), at anchors
    H = [1, 2, 4, 8, 16, 32, 64, 128, 240], log-interpolated. Centring was
    the key fix: v1's uncentred 72h empirical distribution (0.478) was
    dominated by the long-run trend and dead volatility regimes.
  - The spread is SHRUNK x0.6 (a single jump in the window otherwise inflates
    the tails: every overlapping h-minute window containing it inherits it)
    and ADAPTIVELY SCALED by the recent 120-min mean |1-min move| over the
    12h mean |1-min move|, clipped to [0.5, 2.0] (widens when the market is
    currently active, narrows when quiet).
  - Rejected by the grid: momentum in the median (hurts), Gaussian MAD sigma
    (0.390), sum-to-one median (0.314), per-tau shrink, 6h window, winsorising.
  - All quantiles clamped to [0, 1], rounded to 6 decimals, non-decreasing.

Practice loss: 0.281947 (persistence baseline 0.363589; dev board top 0.256,
rank-100 0.275). Honest read: competitive (beats the house baseline by 22%),
top-half of the field, but a long shot for a prize — the gate is the house
superbot's final score, which cannot be measured from here.
"""

import json
import math
import os
from pathlib import Path

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
ANCHORS = [1, 2, 4, 8, 16, 32, 64, 128, 240]
WINDOW_MINUTES = 720
SHRINK = 0.6
HALF_LIFE_MIN = 120.0


def _quantile(sorted_vals, tau):
    """Linear-interpolation quantile of a sorted list (numpy 'linear' method)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def _mean_abs_move(prices):
    if len(prices) < 2:
        return 0.0
    return sum(abs(b - a) for a, b in zip(prices, prices[1:])) / (len(prices) - 1)


def forecast_market(prices, horizon, fair):
    """Return horizon rows x 5 quantiles for one market's price list.

    `fair` is the market's fair value (last price / sum of last prices).
    """
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    recent = prices[-WINDOW_MINUTES:]
    n = len(recent)
    anchor_q = {}
    for h in ANCHORS:
        if h >= n:
            continue
        diffs = sorted(recent[t + h] - recent[t] for t in range(n - h))
        med = _quantile(diffs, 0.5)
        q = [_quantile(diffs, tau) - med for tau in QUANTILES]
        q[2] = 0.0
        anchor_q[h] = q
    if not anchor_q:
        return [[_price6(x)] * len(QUANTILES) for _ in range(horizon)]
    hs = sorted(anchor_q)
    # Adaptive scale: how active is the market right now vs its 12h average?
    base_vol = _mean_abs_move(recent)
    short_vol = _mean_abs_move(prices[-120:])
    scale = 1.0
    if base_vol > 0:
        scale = max(0.5, min(2.0, short_vol / base_vol))
    rows = []
    for h in range(1, horizon + 1):
        if h <= hs[0]:
            qs = anchor_q[hs[0]]
        elif h >= hs[-1]:
            qs = anchor_q[hs[-1]]
        else:
            lo = max(a for a in hs if a <= h)
            hi = min(a for a in hs if a >= h)
            if lo == hi:
                qs = anchor_q[lo]
            else:
                w = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
                qs = [anchor_q[lo][j] * (1.0 - w) + anchor_q[hi][j] * w
                      for j in range(len(QUANTILES))]
        med = fair + (x - fair) * math.exp(-h / HALF_LIFE_MIN)
        row = [med + d * SHRINK * scale if j != 2 else med
               for j, d in enumerate(qs)]
        row = [_price6(v) for v in row]
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_case(case):
    """Forecast every market of a case; returns {id: rows}."""
    horizon = case["horizon_minutes"]
    lasts = {m["id"]: m["prices"][-1] for m in case["markets"]}
    total = sum(lasts.values())
    out = {}
    for m in case["markets"]:
        fair = lasts[m["id"]] / total if total > 0 else lasts[m["id"]]
        out[m["id"]] = forecast_market(m["prices"], horizon, fair)
    return out


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in input_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text(encoding="utf-8"))
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"{path.name} asks for quantiles {case['quantiles']}")
        forecasts = forecast_case(case)
        ordered = [{"id": m["id"], "q": forecasts[m["id"]]} for m in case["markets"]]
        (output_dir / f"{case_id}.out").write_text(
            json.dumps({"forecasts": ordered}) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
