"""
Weekly football prediction bot - European top 5 leagues.
Model: Poisson (time-weighted attack/defence strength).
Data: football-data.co.uk (free, no key) for results, fixtures, odds.
Optional: API-Football key for injury adjustment.
Output: predictions.md, predictions.csv, optional Telegram message.
"""
import io
import os
import difflib
import datetime as dt

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

# ---------- SETTINGS ----------
LEAGUES = {
    # football-data code: (name, API-Football league id)
    "E0": ("Premier League", 39),
    "SP1": ("La Liga", 140),
    "D1": ("Bundesliga", 78),
    "I1": ("Serie A", 135),
    "F1": ("Ligue 1", 61),
}
SEASONS_BACK = 3
HALF_LIFE_DAYS = 180
DAYS_AHEAD = 7
MIN_VALUE = 0.05        # flag bets with 5%+ edge
MAX_GOALS = 10
SHRINK = 3
INJ_ATTACK_PENALTY = 0.02   # own xG down 2% per injured player (guess, not backtested)
INJ_ATTACK_CAP = 0.10
INJ_DEFENCE_PENALTY = 0.01  # opponent xG up 1% per injured player (guess)
INJ_DEFENCE_CAP = 0.05
BASE = "https://www.football-data.co.uk"
# ------------------------------


def season_start_year():
    t = dt.date.today()
    return t.year if t.month >= 8 else t.year - 1


def season_codes(n):
    s = season_start_year()
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(s - n + 1, s + 1)]


def read_csv(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), encoding_errors="ignore")


def load_results(league):
    frames = []
    for code in season_codes(SEASONS_BACK):
        url = f"{BASE}/mmz4281/{code}/{league}.csv"
        try:
            frames.append(read_csv(url))
        except Exception as e:
            print("Skipped", url, e)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"])


def shrunk(values, weights, prior):
    return (np.sum(values * weights) + prior * SHRINK) / (np.sum(weights) + SHRINK)


def fit(df):
    age = (pd.Timestamp.today() - df["Date"]).dt.days.clip(lower=0)
    df = df.assign(w=0.5 ** (age / HALF_LIFE_DAYS))
    avg_h = np.average(df.FTHG, weights=df.w)
    avg_a = np.average(df.FTAG, weights=df.w)
    s = {}
    for t in set(df.HomeTeam) | set(df.AwayTeam):
        h = df[df.HomeTeam == t]
        a = df[df.AwayTeam == t]
        s[t] = {
            "att_h": shrunk(h.FTHG, h.w, avg_h) / avg_h,
            "def_h": shrunk(h.FTAG, h.w, avg_a) / avg_a,
            "att_a": shrunk(a.FTAG, a.w, avg_a) / avg_a,
            "def_a": shrunk(a.FTHG, a.w, avg_h) / avg_h,
        }
    return s, avg_h, avg_a


def fetch_injuries(api_id, known_teams, start, end):
    """Returns {team: number of injured/suspended players} for fixtures in window."""
    key = os.getenv("API_FOOTBALL_KEY")
    if not key:
        return {}
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/injuries",
            headers={"x-apisports-key": key},
            params={"league": api_id, "season": season_start_year()},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("response", [])
    except Exception as e:
        print("Injuries skipped:", e)
        return {}
    seen = {}
    for item in data:
        try:
            d = pd.to_datetime(item["fixture"]["date"], utc=True).tz_convert(None)
        except Exception:
            continue
        if not (start <= d <= end + pd.Timedelta(days=1)):
            continue
        m = difflib.get_close_matches(item["team"]["name"], known_teams, n=1, cutoff=0.5)
        if m:
            seen.setdefault(m[0], set()).add(item["player"]["id"])
    return {t: len(p) for t, p in seen.items()}


def match_probs(home, away, s, avg_h, avg_a, inj_h=0, inj_a=0):
    neutral = {"att_h": 1, "def_h": 1, "att_a": 1, "def_a": 1}
    H = s.get(home, neutral)
    A = s.get(away, neutral)
    xg_h = avg_h * H["att_h"] * A["def_a"]
    xg_a = avg_a * A["att_a"] * H["def_h"]
    # crude injury adjustment
    xg_h *= (1 - min(INJ_ATTACK_PENALTY * inj_h, INJ_ATTACK_CAP)) * \
            (1 + min(INJ_DEFENCE_PENALTY * inj_a, INJ_DEFENCE_CAP))
    xg_a *= (1 - min(INJ_ATTACK_PENALTY * inj_a, INJ_ATTACK_CAP)) * \
            (1 + min(INJ_DEFENCE_PENALTY * inj_h, INJ_DEFENCE_CAP))
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, xg_h), poisson.pmf(g, xg_a))
    m = m / m.sum()
    return {
        "xg_h": xg_h,
        "xg_a": xg_a,
        "p_home": np.tril(m, -1).sum(),
        "p_draw": np.trace(m),
        "p_away": np.triu(m, 1).sum(),
        "p_over25": sum(m[i, j] for i in g for j in g if i + j > 2),
        "p_btts": m[1:, 1:].sum(),
    }


def get_odds(row, cols):
    for c in cols:
        if c in row and pd.notna(row[c]):
            return float(row[c])
    return None


def main():
    try:
        all_fix = read_csv(f"{BASE}/fixtures.csv")
        all_fix["Date"] = pd.to_datetime(all_fix["Date"], dayfirst=True, errors="coerce")
    except Exception as e:
        print("Could not load fixtures:", e)
        return

    today = pd.Timestamp.today().normalize()
    end = today + pd.Timedelta(days=DAYS_AHEAD)
    rows = []

    for code, (name, api_id) in LEAGUES.items():
        fixtures = all_fix[(all_fix["Div"] == code) &
                           (all_fix["Date"] >= today) &
                           (all_fix["Date"] <= end)].sort_values("Date")
        if fixtures.empty:
            continue
        results = load_results(code)
        if results.empty:
            print("No results for", name)
            continue
        s, avg_h, avg_a = fit(results)
        inj = fetch_injuries(api_id, list(s.keys()), today, end)

        for _, f in fixtures.iterrows():
            hn, an = f["HomeTeam"], f["AwayTeam"]
            ih, ia = inj.get(hn, 0), inj.get(an, 0)
            p = match_probs(hn, an, s, avg_h, avg_a, ih, ia)
            odds = {
                "Home": get_odds(f, ["B365H", "AvgH", "MaxH"]),
                "Draw": get_odds(f, ["B365D", "AvgD", "MaxD"]),
                "Away": get_odds(f, ["B365A", "AvgA", "MaxA"]),
            }
            probs = {"Home": p["p_home"], "Draw": p["p_draw"], "Away": p["p_away"]}
            best, best_val = None, 0
            for k in probs:
                if odds[k]:
                    v = probs[k] * odds[k] - 1
                    if v > best_val:
                        best, best_val = k, v
            is_value = bool(best and best_val >= MIN_VALUE)
            rows.append({
                "league": name,
                "date": f["Date"].date(),
                "match": f"{hn} vs {an}",
                "home_%": round(p["p_home"] * 100),
                "draw_%": round(p["p_draw"] * 100),
                "away_%": round(p["p_away"] * 100),
                "xg": f"{p['xg_h']:.1f}-{p['xg_a']:.1f}",
                "over2.5_%": round(p["p_over25"] * 100),
                "btts_%": round(p["p_btts"] * 100),
                "injured": f"{ih}-{ia}" if inj else "n/a",
                "pick": max(probs, key=probs.get),
                "value_bet": f"{best} @ {odds[best]} (+{best_val * 100:.0f}%)" if is_value else "",
                "_val": best_val if is_value else 0,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.drop(columns=["_val"]).to_csv("predictions.csv", index=False)

    lines = [f"WEEKLY PREDICTIONS ({dt.date.today()})", ""]
    if not rows:
        lines.append("No fixtures found in the next 7 days.")
    else:
        top = sorted([r for r in rows if r["_val"] > 0], key=lambda r: -r["_val"])[:5]
        lines.append("TOP VALUE BETS")
        if top:
            for r in top:
                lines.append(f"- {r['match']} ({r['league']}): {r['value_bet']}")
        else:
            lines.append("- None this week. Skip betting.")
        lines.append("")
        for league in LEAGUES.values():
            block = [r for r in rows if r["league"] == league[0]]
            if not block:
                continue
            lines.append(f"== {league[0].upper()} ==")
            for r in block:
                lines.append(
                    f"{r['date']} | {r['match']}\n"
                    f"  H {r['home_%']}% D {r['draw_%']}% A {r['away_%']}% | "
                    f"xG {r['xg']} | O2.5 {r['over2.5_%']}% | BTTS {r['btts_%']}% | "
                    f"Inj {r['injured']}\n"
                    f"  Pick: {r['pick']}" + (f" | VALUE: {r['value_bet']}" if r["value_bet"] else "")
                )
            lines.append("")
    lines.append("Not financial advice. Stake 1-2% max. Check live odds before betting.")
    text = "\n".join(lines)

    with open("predictions.md", "w") as fh:
        fh.write(text)
    print(text)

    token = os.getenv("TELEGRAM_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        for i in range(0, len(text), 3500):
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat, "text": text[i:i + 3500]},
                timeout=30,
            )


if __name__ == "__main__":
    main()
