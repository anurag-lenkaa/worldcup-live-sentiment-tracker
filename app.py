"""
World Cup Live Sentiment Tracker
================================
Real-time pipeline: ingest tweets -> classify sentiment -> aggregate over a
rolling window -> overlay goal events -> live Dash dashboard.

Runs out of the box in REPLAY mode (no API key, no cost): it generates realistic
match chatter whose tone shifts around goal events on a repeating loop, and runs
that text through a real sentiment model so the analysis is genuine, not faked.

Two sentiment backends (set SENTIMENT_BACKEND):
  - "transformers"  HF cardiffnlp roberta. High quality. ~1GB+ RAM. Local / paid host.
  - "vader"         rule-based, tiny, instant, no torch. Fits a 512MB free dyno.

Config is via environment variables so the same code runs locally and on a host.

Local:
    pip install -r requirements-full.txt      # includes torch + transformers
    python app.py                             # http://127.0.0.1:8050

Deploy (gunicorn serves `server`; see README):
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

# --------------------------------------------------------------------------- #
# Config (env-overridable)
# --------------------------------------------------------------------------- #
SOURCE = os.environ.get("SOURCE", "replay")                  # replay | x | bluesky
SENTIMENT_BACKEND = os.environ.get("SENTIMENT_BACKEND", "transformers")  # transformers | vader
MODEL_NAME = os.environ.get("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
MATCH_QUERY = os.environ.get("MATCH_QUERY", "World Cup OR Mexico OR South Africa")
PORT = int(os.environ.get("PORT", 8050))
ROLLING_WINDOW_SEC = 60          # smoothing window for the net-sentiment line
MAX_BUFFER = 5000                # max tweets kept in memory
DASH_REFRESH_MS = 2000           # how often the UI repolls the buffer

# Replay schedule: goals recur every LOOP_MINUTES so the demo stays lively.
KICKOFF = datetime.utcnow()
LOOP_MINUTES = 5.0
GOAL_SCHEDULE = [(0.5, "Mexico"), (2.0, "South Africa"), (3.5, "Mexico")]

# --------------------------------------------------------------------------- #
# Shared in-memory stores
# --------------------------------------------------------------------------- #
BUFFER = deque(maxlen=MAX_BUFFER)        # {"ts","text","label","score"}
BUFFER_LOCK = threading.Lock()
GOAL_LOG = deque(maxlen=50)              # {"ts","label"} actual goal timestamps
GOAL_LOCK = threading.Lock()


def log_goal(team):
    """Record a goal at the current time. Replay calls this; for a live match,
    call it from your sports-feed poller so the red lines mark real goals."""
    with GOAL_LOCK:
        GOAL_LOG.append({"ts": datetime.utcnow(), "label": f"GOAL {team}"})


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
class SentimentAnalyzer:
    """Two backends behind one interface."""

    def __init__(self, backend=SENTIMENT_BACKEND, model_name=MODEL_NAME):
        self.backend = backend
        self._model_name = model_name
        self._engine = None

    def _load(self):
        if self.backend == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._engine = SentimentIntensityAnalyzer()
            print("[sentiment] loaded VADER")
        else:
            from transformers import pipeline
            self._engine = pipeline("sentiment-analysis", model=self._model_name)
            print(f"[sentiment] loaded {self._model_name}")

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
# Tweet sources (pluggable)
# --------------------------------------------------------------------------- #
class ReplaySource:
    """Synthetic but realistic chatter. Tone skews toward whichever team just
    scored, so the dashboard shows a genuine swing on each goal — and the text is
    still classified by the real model, so the sentiment output is honest."""

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

    def __init__(self, teams=("Mexico", "South Africa")):
        self.teams = teams
        self._fired = set()

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
            time.sleep(random.uniform(0.15, 0.5))


class XApiSource:
    """X API v2 filtered stream via Tweepy's modern StreamingClient.
    Requires PAID access (pay-per-use ~$0.005/read in 2026). The v1.1
    AppAuthHandler / api.search pattern from older tutorials no longer works."""

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
    """Free alternative: Bluesky firehose via `atproto`. No paid tier."""

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
# Ingestion worker
# --------------------------------------------------------------------------- #
def ingestion_loop():
    analyzer = SentimentAnalyzer()
    source = make_source()
    print(f"[ingest] source={SOURCE} backend={SENTIMENT_BACKEND}")
    for text in source.stream():
        if not text:
            continue
        try:
            label, score = analyzer.classify(text)
        except Exception as exc:
            print(f"[ingest] classify error: {exc}")
            continue
        with BUFFER_LOCK:
            BUFFER.append({"ts": datetime.utcnow(), "text": text,
                           "label": label, "score": score})


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #
app = dash.Dash(__name__)
server = app.server          # <-- gunicorn entry point (gunicorn app:server)
app.title = "WC Live Sentiment"

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "maxWidth": "980px",
           "margin": "0 auto", "padding": "16px"},
    children=[
        html.H2("World Cup - Live Sentiment Tracker"),
        html.Div(id="headline", style={"fontSize": "15px", "color": "#555"}),
        dcc.Graph(id="timeline"),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            dcc.Graph(id="distribution", style={"flex": "1 1 320px"}),
            html.Div(style={"flex": "1 1 320px"}, children=[
                html.H4("Latest reactions"),
                html.Div(id="recent"),
            ]),
        ]),
        dcc.Interval(id="tick", interval=DASH_REFRESH_MS, n_intervals=0),
    ],
)


def _snapshot():
    with BUFFER_LOCK:
        return list(BUFFER)


@app.callback(
    [Output("timeline", "figure"), Output("distribution", "figure"),
     Output("recent", "children"), Output("headline", "children")],
    [Input("tick", "n_intervals")],
)
def refresh(_):
    data = _snapshot()
    if not data:
        empty = go.Figure()
        return empty, empty, "Waiting for tweets...", "Starting up - loading model..."

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
    fig_time.add_trace(go.Scatter(x=times, y=smoothed, mode="lines",
                                  name="net sentiment", line={"width": 3}))
    fig_time.add_hline(y=0, line_dash="dot", line_color="#aaa")
    with GOAL_LOCK:
        goals = list(GOAL_LOG)
    for ev in goals:
        if times[0] <= ev["ts"] <= times[-1] + timedelta(seconds=5):
            fig_time.add_vline(x=ev["ts"], line_color="#e63946", line_dash="dash")
            fig_time.add_annotation(x=ev["ts"], y=1, text=ev["label"],
                                    showarrow=False, font={"size": 10})
    fig_time.update_layout(title="Net sentiment over time (1 = all positive)",
                           yaxis_range=[-1.05, 1.15], margin=dict(t=40, b=20), height=340)

    counts = {"POS": 0, "NEU": 0, "NEG": 0}
    for d in data:
        counts[d["label"]] += 1
    fig_dist = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[counts["POS"], counts["NEU"], counts["NEG"]],
        marker={"colors": ["#2a9d8f", "#e9c46a", "#e76f51"]}, hole=0.45))
    fig_dist.update_layout(title="Sentiment mix", height=300, margin=dict(t=40, b=10))

    color = {"POS": "#2a9d8f", "NEU": "#b08900", "NEG": "#e76f51"}
    recent = [
        html.Div(style={"borderLeft": f"4px solid {color[d['label']]}",
                        "padding": "4px 8px", "margin": "4px 0", "fontSize": "13px"},
                 children=f"[{d['label']}] {d['text']}")
        for d in data[-8:][::-1]
    ]

    total = len(data)
    headline = (f"{total} reactions analysed - {100*counts['POS']/total:.0f}% positive "
                f"- source: {SOURCE} - backend: {SENTIMENT_BACKEND}")
    return fig_time, fig_dist, recent, headline


# Start ingestion at import time so it also runs under gunicorn (single worker).
# Tests set DISABLE_INGESTION=1 to skip it.
if os.environ.get("DISABLE_INGESTION") != "1":
    threading.Thread(target=ingestion_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
