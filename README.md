# World Cup — Live Sentiment Tracker

A real-time pipeline that ingests match-day social chatter, classifies sentiment,
and visualizes how public mood swings around goals on a live dashboard.

**Stack:** Python · Dash + Plotly (live UI) · Hugging Face Transformers
(`cardiffnlp/twitter-roberta-base-sentiment-latest`) · pluggable streaming source.

It runs out of the box with **zero API keys and zero cost** in replay mode: it
generates realistic match chatter whose tone shifts around recurring goal events
and classifies that text with a real model, so the sentiment line genuinely
reacts to "goals." Swap in a live feed when you want the real thing.

---

## Quick start (local)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements-full.txt               # includes torch + transformers
python app.py                                       # open http://127.0.0.1:8050
```

First run downloads the model (~500MB) once and caches it.

---

## Two sentiment backends

| Backend         | Quality | RAM     | Set with                          |
|-----------------|---------|---------|-----------------------------------|
| `transformers`  | High    | ~1GB+   | `SENTIMENT_BACKEND=transformers`  |
| `vader`         | Decent  | tiny    | `SENTIMENT_BACKEND=vader`         |

Use `transformers` locally (and on a 2GB+ host). Use `vader` on a 512MB free dyno
so the live demo stays up without OOM. The repo defaults to `transformers`;
`render.yaml` overrides it to `vader` for the free deploy.

---

## Deploy live on Render (free)

1. Push this repo to GitHub (see below).
2. On Render: **New → Blueprint**, point it at the repo. `render.yaml` is detected
   automatically and provisions a free web service running gunicorn + VADER.
3. You get an HTTPS URL. Done.

Notes on the free tier:
- It sleeps after ~15 min idle, so the first hit after a quiet spell takes
  30–60s to cold-start, then fills in over a few seconds.
- 512MB RAM is why the deploy uses VADER, not torch.

**To run the actual transformer live:** use Render's Standard plan (2GB, ~$25/mo),
swap the build to `pip install -r requirements-full.txt`, and set
`SENTIMENT_BACKEND=transformers`.

---

## Switching to real tweets

Set `SOURCE` and provide credentials.

- `SOURCE=bluesky` — free firehose via `atproto`. Recommended: no paid tier.
  Uncomment `atproto` in `requirements-full.txt`.
- `SOURCE=x` — X API v2 filtered stream via Tweepy `StreamingClient`. Requires
  **paid** access (pay-per-use ~$0.005 per post read in 2026; the old free v1.1
  `api.search` pattern no longer works). Set `X_BEARER_TOKEN`. Uncomment `tweepy`.

### Goal events for a real match
In replay, goals are logged automatically. For a live match, poll a sports feed
for goal timestamps and call `log_goal("Mexico")` from your poller — the red lines
on the timeline are driven entirely by `GOAL_LOG`.

---

## Architecture

```
source.stream() ──► ingestion thread ──► classify ──► BUFFER (deque)
                                                         │
                          GOAL_LOG (goal timestamps) ────┤
                                                         ▼
                          Dash callback (polls every 2s) renders:
                          net-sentiment timeline + goal lines, mix pie, live feed
```

The streaming source, the sentiment engine, and the visualization are independent
— each is swappable without touching the others. Under gunicorn the app runs a
single worker so all requests share one in-memory buffer and one ingestion thread
(do not enable `--preload` or multiple workers without externalizing state to,
e.g., Redis).

---

## Push to GitHub

```bash
git init
git add .
git commit -m "World Cup live sentiment tracker"
git branch -M main
git remote add origin https://github.com/<you>/wc-sentiment.git
git push -u origin main
```
