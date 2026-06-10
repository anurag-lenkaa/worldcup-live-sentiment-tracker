"""
World Cup Live Sentiment Tracker
================================
Real-time pipeline: ingest tweets -> classify sentiment -> aggregate over a
rolling window -> overlay goal events -> live Dash dashboard.

Runs out of the box in REPLAY mode (no API key, no cost). Two sentiment backends
(SENTIMENT_BACKEND): "transformers" (HF roberta, ~1GB RAM) or "vader" (tiny, fits
a 512MB free dyno). Config is via environment variables.

Local:
    pip install -r requirements-full.txt
    python app.py                       # http://127.0.0.1:8050
Deploy:
    gunicorn app:server --workers 1 --threads 8 --timeout 120
"""

import os
import random
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Import the light engine in the MAIN thread at module load. Importing inside a
# worker thread that was spawned during module import can deadlock on Python's
# import lock (race only visible on slow CPUs, e.g. free-tier dynos).
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderEngine
except ImportError:
    _VaderEngine = None

# --------------------------------------------------------------------------- #
# Config (env-overridable)
# --------------------------------------------------------------------------- #
SOURCE = os.environ.get("SOURCE", "replay")                  # replay | x | bluesky
SENTIMENT_BACKEND = os.environ.get("SENTIMENT_BACKEND", "transformers")  # transformers | vader
MODEL_NAME = os.environ.get("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
MATCH_QUERY = os.environ.get("MATCH_QUERY", "World Cup OR Mexico OR South Africa")
PORT = int(os.environ.get("PORT", 8050))
ROLLING_WINDOW_SEC = 60
MAX_BUFFER = 5000
DASH_REFRESH_MS = 2000

KICKOFF = datetime.utcnow()
LOOP_MINUTES = 5.0
GOAL_SCHEDULE = [(0.5, "Mexico"), (2.0, "South Africa"), (3.5, "Mexico")]

# --------------------------------------------------------------------------- #
# Shared state
# --------------------------------------------------------------------------- #
BUFFER = deque(maxlen=MAX_BUFFER)        # {"ts","text","label","score"}
BUFFER_LOCK = threading.Lock()
GOAL_LOG = deque(maxlen=50)              # {"ts","label"}
GOAL_LOCK = threading.Lock()
# Self-diagnostics surfaced on the page so you never need to dig through logs.
STATUS = {"started": False, "stage": "idle", "attempts": 0, "errors": 0,
          "last_error": "", "beat": 0.0}


def log_goal(team):
    with GOAL_LOCK:
        GOAL_LOG.append({"ts": datetime.utcnow(), "label": f"GOAL {team}"})


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
class SentimentAnalyzer:
    def __init__(self, backend=SENTIMENT_BACKEND, model_name=MODEL_NAME):
        self.backend = backend
        self._model_name = model_name
        self._engine = None

    def _load(self):
        if self.backend == "vader":
            if _VaderEngine is None:
                raise RuntimeError("vaderSentiment is not installed")
            self._engine = _VaderEngine()
            print("[sentiment] loaded VADER", flush=True)
        else:
            from transformers import pipeline
            self._engine = pipeline("sentiment-analysis", model=self._model_name)
            print(f"[sentiment] loaded {self._model_name}", flush=True)

    def classify(self, text):
        if self._engine is None:
            self._load()
        text = text[:280]
        if self.backend == "vader":
            c = self._engine.polarity_scores(text)["compound"]
            if c >= 0.05:
                return "POS", abs(c)
            if c <= -0.05:
                return "NEG", abs(c)
            return "NEU", 1 - abs(c)
        out = self._engine(text)[0]
        label = out["label"].lower()
        label = "POS" if label.startswith("pos") else "NEG" if label.startswith("neg") else "NEU"
        return label, float(out["score"])


# --------------------------------------------------------------------------- #
# Tweet sources
# --------------------------------------------------------------------------- #
class ReplaySource:
    POS = [
        "What a goal!! {team} are unstoppable today, absolutely buzzing",
        "{team} deserve this, brilliant football, so proud right now",
        "GET IN!!! {team} take the lead, this team is special",
        "Beautiful move from {team}, world class finish",
    ]
    NEG = [
        "Awful defending from {team}, this is embarrassing honestly",
        "How did {team} concede that?? Heartbreaking, gutted",
        "{team} are a disaster, the manager has no plan, terrible",
        "Cannot believe {team} let that in, furious right now",
    ]
    NEU = [
        "{team} have a corner coming up, lets see",
        "Halftime stats look even between both sides",
        "Watching {team} with the family, decent atmosphere",
        "Anyone know the lineup change for {team}?",
    ]

    def __init__(self, teams=("Mexico", "South Africa"), burst=30):
        self.teams = teams
        self._fired = set()
        self._burst = burst

    def _mood_for_now(self):
        now = datetime.utcnow()
        elapsed = (now - KICKOFF).total_seconds() / 60.0
        loop_idx = int(elapsed // LOOP_MINUTES)
        local = elapsed % LOOP_MINUTES
        for i, (gm, team) in enumerate(GOAL_SCHEDULE):
            if gm <= local <= gm + 0.5:
                key = (loop_idx, i)
                if key not in self._fired:
                    self._fired.add(key)
                    log_goal(team)
                return team, "pos"
        return random.choice(self.teams), "neutral"

    def stream(self):
        while True:
            team, bias = self._mood_for_now()
            if bias == "pos":
                if random.random() < 0.7:
                    tmpl, subj = random.choice(self.POS), team
                else:
                    other = [t for t in self.teams if t != team][0]
                    tmpl, subj = random.choice(self.NEG), other
            else:
                tmpl = random.choice(self.POS + self.NEG + self.NEU)
                subj = random.choice(self.teams)
            yield tmpl.format(team=subj)
            if self._burst > 0:
                self._burst -= 1          # instant prime so the page fills fast
            else:
                time.sleep(random.uniform(0.15, 0.5))


class XApiSource:
    """X API v2 filtered stream via Tweepy StreamingClient. Requires PAID access."""

    def __init__(self, bearer_token, query=MATCH_QUERY):
        self.bearer_token = bearer_token
        self.query = query

    def stream(self):
        import tweepy
        bucket = deque()

        class _Listener(tweepy.StreamingClient):
            def on_tweet(self, tweet):
                bucket.append(tweet.text)

        client = _Listener(self.bearer_token)
        existing = client.get_rules()
        if existing and existing.data:
            client.delete_rules([r.id for r in existing.data])
        client.add_rules(tweepy.StreamRule(f"({self.query}) lang:en -is:retweet"))
        threading.Thread(target=lambda: client.filter(tweet_fields=["created_at"]),
                         daemon=True).start()
        while True:
            yield bucket.popleft() if bucket else time.sleep(0.1)


class BlueskySource:
    """Free firehose via atproto. No paid tier."""

    def __init__(self, keywords=("world cup", "mexico", "south africa")):
        self.keywords = [k.lower() for k in keywords]

    def stream(self):
        from atproto import (FirehoseSubscribeReposClient,
                             parse_subscribe_repos_message, CAR, models)
        bucket = deque()

        def on_message(message):
            commit = parse_subscribe_repos_message(message)
            if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
                return
            car = CAR.from_bytes(commit.blocks)
            for op in commit.ops:
                if op.action == "create" and op.cid:
                    rec = car.blocks.get(op.cid) or {}
                    text = rec.get("text", "")
                    if text and any(k in text.lower() for k in self.keywords):
                        bucket.append(text)

        client = FirehoseSubscribeReposClient()
        threading.Thread(target=lambda: client.start(on_message), daemon=True).start()
        while True:
            yield bucket.popleft() if bucket else time.sleep(0.1)


def make_source():
    if SOURCE == "x":
        return XApiSource(bearer_token=os.environ["X_BEARER_TOKEN"])
    if SOURCE == "bluesky":
        return BlueskySource()
    return ReplaySource()


# --------------------------------------------------------------------------- #
# Ingestion worker + lazy starter
# --------------------------------------------------------------------------- #
def ingestion_loop():
    STATUS["started"] = True
    STATUS["beat"] = time.time()
    print(f"[ingest] source={SOURCE} backend={SENTIMENT_BACKEND}", flush=True)
    try:
        STATUS["stage"] = "loading engine"
        analyzer = SentimentAnalyzer()
        analyzer.classify("warm up engine now")     # force load up front
        STATUS["stage"] = "engine ready"
        STATUS["beat"] = time.time()
        source = make_source()
        STATUS["stage"] = "streaming"
        for text in source.stream():
            STATUS["beat"] = time.time()
            if not text:
                continue
            STATUS["attempts"] += 1
            try:
                label, score = analyzer.classify(text)
            except Exception as exc:
                STATUS["errors"] += 1
                STATUS["last_error"] = str(exc)
                continue
            with BUFFER_LOCK:
                BUFFER.append({"ts": datetime.utcnow(), "text": text,
                               "label": label, "score": score})
    except BaseException as exc:                    # NOTHING dies silently
        STATUS["stage"] = "crashed"
        STATUS["last_error"] = f"{exc.__class__.__name__}: {exc}"
        print(f"[ingest] FATAL: {STATUS['last_error']}", flush=True)


_start_lock = threading.Lock()


def ensure_ingestion():
    """Idempotent: guarantees one ingestion thread runs in THIS process. Called
    both at import and from the callback, so it works regardless of how the host
    forks workers."""
    with _start_lock:
        if not STATUS["started"]:
            STATUS["started"] = True  # set immediately to avoid a race
            threading.Thread(target=ingestion_loop, daemon=True).start()


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #
FONTS = "https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&display=swap"
app = dash.Dash(__name__, external_stylesheets=[FONTS])
server = app.server          # gunicorn entry point: gunicorn app:server
app.title = "WC Live Sentiment"
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  :root{
    --bg:#0A1422; --panel:#101E33; --panel2:#0D1929; --line:#1C3050;
    --ink:#EAF2FF; --ink-dim:#8DA2C0;
    --pos:#27E07D; --neg:#FF5A6E; --neu:#F4B942;
  }
  html,body{margin:0;background:
    radial-gradient(1200px 500px at 70% -10%, #14304F 0%, transparent 60%),
    radial-gradient(900px 400px at 10% 110%, #0F2A3F 0%, transparent 55%),
    var(--bg);
    color:var(--ink); font-family:Inter,system-ui,sans-serif;}
  .wrap{max-width:1020px;margin:0 auto;padding:28px 20px 48px;}
  .topbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px;}
  .brand{font-family:Oswald,sans-serif;font-weight:700;font-size:30px;letter-spacing:.04em;
    text-transform:uppercase;}
  .brand .accent{color:var(--pos);}
  .live{display:inline-flex;align-items:center;gap:8px;font-family:Oswald,sans-serif;
    font-weight:600;font-size:13px;letter-spacing:.18em;color:#FF6B7C;
    border:1px solid #FF5A6E55;border-radius:999px;padding:5px 14px;background:#FF5A6E12;}
  .live .dot{width:8px;height:8px;border-radius:50%;background:#FF5A6E;
    animation:pulse 1.4s ease-in-out infinite;}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(.75)}}
  @media (prefers-reduced-motion: reduce){.live .dot{animation:none}}
  #headline{font-size:13px;color:var(--ink-dim);letter-spacing:.02em;margin:6px 0 18px;}
  .panel{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
    border:1px solid var(--line);border-radius:16px;padding:10px 12px 4px;
    box-shadow:0 8px 28px #00000040;}
  .row{display:flex;gap:16px;flex-wrap:wrap;margin-top:16px;}
  .ticker-title{font-family:Oswald,sans-serif;font-weight:600;font-size:14px;
    letter-spacing:.14em;text-transform:uppercase;color:var(--ink-dim);
    padding:14px 14px 8px;}
  #recent{padding:0 12px 12px;}
  #recent > div{border-left-width:3px !important;border-radius:0 8px 8px 0;
    background:#FFFFFF08;color:var(--ink);}
</style>
{%scripts%}
</head>
<body>{%app_entry%}<footer>{%config%}{%renderer%}</footer></body>
</html>"""

app.layout = html.Div(className="wrap", children=[
    html.Div(className="topbar", children=[
        html.Div(["WORLD CUP ", html.Span("SENTIMENT", className="accent")], className="brand"),
        html.Div([html.Span(className="dot"), "LIVE"], className="live"),
    ]),
    html.Div(id="headline"),
    html.Div(className="panel", children=[dcc.Graph(id="timeline", config={"displayModeBar": False})]),
    html.Div(className="row", children=[
        html.Div(className="panel", style={"flex": "1 1 320px"},
                 children=[dcc.Graph(id="distribution", config={"displayModeBar": False})]),
        html.Div(className="panel", style={"flex": "1 1 320px"}, children=[
            html.Div("Fan reactions - live ticker", className="ticker-title"),
            html.Div(id="recent"),
        ]),
    ]),
    dcc.Interval(id="tick", interval=DASH_REFRESH_MS, n_intervals=0),
])


def _snapshot():
    with BUFFER_LOCK:
        return list(BUFFER)


@app.callback(
    [Output("timeline", "figure"), Output("distribution", "figure"),
     Output("recent", "children"), Output("headline", "children")],
    [Input("tick", "n_intervals")],
)
def refresh(_):
    ensure_ingestion()                       # start the worker in the serving process
    data = _snapshot()
    if not data:
        empty = go.Figure()
        beat_age = (time.time() - STATUS["beat"]) if STATUS["beat"] else -1
        diag = (f"Warming up \u00b7 stage: {STATUS['stage']} \u00b7 "
                f"{STATUS['attempts']} pulled \u00b7 {STATUS['errors']} errors \u00b7 "
                f"heartbeat {beat_age:.0f}s ago \u00b7 {SOURCE}/{SENTIMENT_BACKEND}"
                + (f" \u00b7 last error: {STATUS['last_error']}" if STATUS["last_error"] else ""))
        return empty, empty, "Waiting for tweets...", diag

    score_map = {"POS": 1, "NEU": 0, "NEG": -1}
    buckets = {}
    for d in data:
        buckets.setdefault(d["ts"].replace(microsecond=0), []).append(score_map[d["label"]])
    times = sorted(buckets)
    net = [sum(buckets[t]) / len(buckets[t]) for t in times]
    smoothed = []
    for i, t in enumerate(times):
        win = [net[j] for j, tj in enumerate(times)
               if 0 <= (t - tj).total_seconds() <= ROLLING_WINDOW_SEC]
        smoothed.append(sum(win) / len(win))

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=times, y=smoothed, mode="lines", name="net sentiment",
        line={"width": 3, "color": "#27E07D", "shape": "spline", "smoothing": 0.6},
        fill="tozeroy", fillcolor="rgba(39,224,125,0.10)"))
    fig_time.add_hline(y=0, line_dash="dot", line_color="#3A4F70")
    with GOAL_LOCK:
        goals = list(GOAL_LOG)
    for ev in goals:
        if times[0] <= ev["ts"] <= times[-1] + timedelta(seconds=5):
            fig_time.add_vline(x=ev["ts"], line_color="#FF5A6E", line_dash="dash")
            fig_time.add_annotation(x=ev["ts"], y=1.08, text="\u26bd " + ev["label"],
                                    showarrow=False, font={"size": 11, "color": "#FF8C9A",
                                                           "family": "Oswald, sans-serif"},
                                    bgcolor="#FF5A6E1A", borderpad=3)
    fig_time.update_layout(
        title={"text": "NET CROWD SENTIMENT", "font": {"family": "Oswald, sans-serif",
               "size": 15, "color": "#8DA2C0"}, "x": 0.02},
        yaxis_range=[-1.05, 1.2], margin=dict(t=46, b=24, l=36, r=18), height=340,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#8DA2C0", "size": 11},
        xaxis={"gridcolor": "#1C3050", "zerolinecolor": "#1C3050"},
        yaxis={"gridcolor": "#1C3050", "zerolinecolor": "#3A4F70",
               "tickvals": [-1, 0, 1],
               "ticktext": ["all negative", "neutral", "all positive"]},
        showlegend=False)

    counts = {"POS": 0, "NEU": 0, "NEG": 0}
    for d in data:
        counts[d["label"]] += 1
    fig_dist = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[counts["POS"], counts["NEU"], counts["NEG"]],
        marker={"colors": ["#27E07D", "#F4B942", "#FF5A6E"],
                "line": {"color": "#0A1422", "width": 3}},
        hole=0.62, textinfo="percent",
        textfont={"family": "Oswald, sans-serif", "size": 13, "color": "#EAF2FF"}))
    fig_dist.add_annotation(text=f"<b>{counts['POS']+counts['NEU']+counts['NEG']}</b><br>reactions",
                            showarrow=False, font={"size": 14, "color": "#EAF2FF",
                                                   "family": "Oswald, sans-serif"})
    fig_dist.update_layout(
        title={"text": "SENTIMENT MIX", "font": {"family": "Oswald, sans-serif",
               "size": 15, "color": "#8DA2C0"}, "x": 0.02},
        height=300, margin=dict(t=46, b=14),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"color": "#8DA2C0", "family": "Inter, sans-serif"},
                "orientation": "h", "y": -0.08})

    color = {"POS": "#27E07D", "NEU": "#F4B942", "NEG": "#FF5A6E"}
    recent = [
        html.Div(style={"borderLeft": f"3px solid {color[d['label']]}",
                        "padding": "6px 10px", "margin": "6px 0", "fontSize": "13px",
                        "lineHeight": "1.45"},
                 children=d["text"])
        for d in data[-8:][::-1]
    ]

    total = len(data)
    headline = (f"{total} reactions analysed  \u00b7  {100*counts['POS']/total:.0f}% positive  "
                f"\u00b7  source: {SOURCE}  \u00b7  engine: {SENTIMENT_BACKEND}")
    return fig_time, fig_dist, recent, headline


# NOTE: ingestion is deliberately NOT started at import time. Spawning a thread
# during module import can deadlock on the import lock under gunicorn (seen on
# slow free-tier CPUs). The first Dash callback calls ensure_ingestion(), which
# is guaranteed to run after imports complete.

if __name__ == "__main__":
    if os.environ.get("DISABLE_INGESTION") != "1":
        ensure_ingestion()
    app.run(debug=False, host="0.0.0.0", port=PORT)
