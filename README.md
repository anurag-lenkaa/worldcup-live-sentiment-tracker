# World Cup — Live Sentiment Tracker

A real-time pipeline that ingests match-day social chatter, classifies sentiment,
and visualizes how crowd mood swings around goals — on a dark, broadcast-style
live dashboard.

**Live demo:** https://wc-sentiment.onrender.com
*(free hosting: first load after idle takes ~30–60s to wake, then fills within seconds)*

**Stack:** Python · Dash + Plotly (live UI) · Hugging Face Transformers
(`cardiffnlp/twitter-roberta-base-sentiment-latest`) locally / VADER on the free
host · pluggable streaming source (replay / X API v2 / Bluesky firehose).

It runs out of the box with **zero API keys and zero cost** in replay mode: it
generates realistic match chatter whose tone shifts around recurring goal events
and classifies that text with a real model — so the sentiment line genuinely
reacts to "goals." On startup it primes itself with an instant burst of ~30
classified reactions, so the dashboard renders full within seconds even on a
cold start.

---

## What you see

A night-match broadcast theme: scoreboard typography, a pulsing LIVE badge, a
pitch-green net-sentiment line over time with goal moments flagged in red, a
sentiment-mix donut, and a color-coded live ticker of the latest fan reactions.
While the buffer is still empty, the header shows a self-diagnostic status line
(pipeline stage, items pulled, errors, heartbeat) instead of a blank screen.

---

## Quick start (local)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements-full.txt               # includes torch + transformers
python app.py                                       # open http://127.0.0.1:8050
```

First run downloads the HF model (~500MB) once and caches it.

---

## Two sentiment backends

| Backend         | Quality | RAM     | Set with                          |
|-----------------|---------|---------|-----------------------------------|
| `transformers`  | High    | ~1GB+   | `SENTIMENT_BACKEND=transformers`  |
| `vader`         | Decent  | tiny    | `SENTIMENT_BACKEND=vader`         |

Use `transformers` locally (and on a 2GB+ host). The free 512MB dyno runs
`vader` so the live demo stays up without OOM — `render.yaml` sets this
automatically. Dependencies are pinned (`dash==2.18.2`) so local and deployed
behavior match.

---

## Deploy on Render (free)

1. Push this repo to GitHub.
2. On Render: **New → Blueprint**, pick the repo. `render.yaml` provisions a
   free web service running `gunicorn app:server` with VADER.
3. You get an HTTPS URL.

To run the actual transformer live, use a 2GB+ plan, build with
`requirements-full.txt`, and set `SENTIMENT_BACKEND=transformers`.

---

## Switching to real tweets

Set `SOURCE` and provide credentials.

`SOURCE=bluesky` — free firehose via `atproto` (recommended; no paid tier).
Uncomment `atproto` in `requirements-full.txt`.

`SOURCE=x` — X API v2 filtered stream via Tweepy `StreamingClient`. Requires
**paid** access (pay-per-use, ~$0.005 per post read as of 2026; the old free
v1.1 `api.search` pattern no longer works). Set `X_BEARER_TOKEN`.

**Goal events for a real match:** in replay, goals log automatically. For a live
match, poll a sports feed and call `log_goal("Mexico")` when a goal lands — the
red flags on the timeline are driven entirely by `GOAL_LOG`.

---

## Architecture

```
source.stream() ──► ingestion thread ──► classify ──► BUFFER (deque)
                                                         │
                          GOAL_LOG (goal timestamps) ────┤
                                                         ▼
                          Dash callback (polls every 2s) renders:
                          sentiment timeline + goal flags, mix donut, live ticker
```

The streaming source, the sentiment engine, and the visualization are
independent — each swaps without touching the others. The app runs a single
gunicorn worker so all requests share one in-memory buffer and one ingestion
thread (externalize state to Redis before scaling workers).

### Production hardening (lessons from deploying this)

The ingestion thread starts **lazily from the first request**, never at module
import. Spawning a thread during import that itself imports packages can
deadlock on Python's import lock — a race that never fired on a fast dev
machine but froze every time on a throttled free-tier CPU. The light engine
(VADER) is imported in the main thread at module load for the same reason.

The whole worker loop is wrapped so nothing dies silently: any fatal error is
captured and surfaced on the page itself, alongside a stage indicator and a
heartbeat. If the dashboard is ever empty, the status line says exactly why.

---

## Troubleshooting

Read the grey status line under the title. `stage: streaming` with the pulled
count climbing means all is well and the page fills momentarily. `stage:
crashed` or a stuck heartbeat shows the exact error text — no log digging
needed. On the free host, an empty page right after a long idle is just the
dyno waking up.
