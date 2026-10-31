# Predictive Race Strategy

An educational F1 strategy lab. A Python model learns lap-time behaviour from
FastF1 and OpenF1 data, simulates dry 1-stop and 2-stop strategies, and serves
results to a Next.js and React dashboard.

## Run locally

### Install once

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
```

### Real race timing sources

The app tries FastF1 first when F1 credentials are present. Otherwise, or if
FastF1 cannot load the selected session, it tries the public OpenF1 API for
historical sessions (2023 onward). OpenF1 historical data needs no account or
API key. Its free API is rate-limited, and live-session data requires an OpenF1
subscription. Responses are cached under `data/raw/openf1` and refreshed after
12 hours. If neither source can supply real laps, the API returns a clear
service-unavailable error; it does not silently substitute synthetic data.
Synthetic laps are available only for explicit UI-development opt-in with
`ALLOW_DEMO_DATA=true`.

To enable FastF1 as the preferred source, enter credentials in the PowerShell
window that starts the app. They are only used by processes started from that
window and are not written to the repository:

```powershell
$credential = Get-Credential -Message "Enter your Formula 1 account credentials"
$env:F1_USERNAME = $credential.UserName
$env:F1_PASSWORD = $credential.GetNetworkCredential().Password
```

For local use, put the credentials in the ignored `.env` file and restart the
Python API. The project loads `.env` at API startup; `.env.example` contains
empty placeholders only. For deployment, configure these variables as secrets
in the Python API service's hosting dashboard instead of deploying a credential
file. GitHub Actions uses repository Actions secrets with the same names.

### Start both services

From the project folder, run this single command:

```powershell
npm run dev:all
```

Open [http://localhost:3000](http://localhost:3000). The Python API documentation
is at [http://localhost:8000/docs](http://localhost:8000/docs).

The year and event selectors use the live FastF1 season calendar, default to the
latest completed Grand Prix, and include a refresh button. Select another season
or event to reload its data. Historical OpenF1 timing is available without F1
credentials from 2023 onward; FastF1 credentials are optional and enable the
FastF1 source to be tried first. If neither provider returns real laps, the
dashboard identifies the deterministic demo data rather than presenting it as
real timing. OpenF1 provides historical data, not live-session timing; live
timing requires an OpenF1 subscription or a working authenticated FastF1
connection. The Next.js same-origin API proxy reports a clear service-unavailable
message if the Python API is stopped; `npm run dev:all` starts both services.

## Tests and checks

```powershell
python -m pytest -q
npm run typecheck
npm run build
```

## Architecture

- `app/` — Next.js App Router, React interface, responsive styling and charts.
- `api/main.py` — FastAPI endpoints for race data and strategy simulations.
- `src/` — FastF1/OpenF1 loading, lap cleaning, feature engineering, model training and simulation.

## Deploy to Render (free, ephemeral)

The root `render.yaml` Blueprint defines two free Docker services in Singapore:
`apexa-api` (FastAPI) and `apexa-web` (Next.js). Deploy from
[Render Blueprints](https://dashboard.render.com/blueprints/new) by selecting
**New → Blueprint**, connecting `VaishnaviNayak2023/Apex`, and deploying the
Blueprint from `main`.

The Blueprint creates a random `DASHBOARD_API_TOKEN` on the API and passes it
to the web service. The API requires this token for its `/api/*` routes except
the public health check; the Next.js server adds it when proxying requests. The
API service URL is configured as `https://apexa-api.onrender.com`. If Render
assigns a different API URL, update the web service's `RACE_API_URL` to that
public API URL. If you want FastF1, add `F1_USERNAME` and `F1_PASSWORD` as
API-service secrets in Render after deployment; public OpenF1 historical data
works without them. Do not send or commit account credentials to GitHub.

**Free-tier limitations:** both services use ephemeral filesystems, so the API
registry, downloaded timing cache, model artifacts, and prediction ledger can
be lost on restart, deploy, or idle spin-down. Render spins free web services
down after 15 minutes without traffic; waking one can take about a minute.
The API is exposed through its public Render URL because free web services
cannot receive private-network requests. Its data-changing/API routes are
protected by the generated service token, and only the health endpoint is
public. This setup is suitable for trying the dashboard, not for durable
production storage or reliable scheduled forecasts.

For retained SQLite/cache/model data, the API needs a paid Render service with
a persistent disk at `/app/data`, or the app needs to be changed to use an
external persistent database/object store. The free Blueprint intentionally
does not create billable resources.

## Race registry, predictions, and scorecards

The lifecycle extension maintains a local SQLite registry covering every race
published by Jolpica-F1 from 1950 through the current season. Current-season
schedule details are refreshed from FastF1 when available; the registry falls
back to the Jolpica calendar with an explicit warning if that refresh fails.
The source-reported calendar is authoritative: cancelled or rescheduled events
are not hard-coded from a static season list. A future event removed from the
upstream schedule is retained locally as `cancelled` rather than silently
deleted. Data tiers follow the project eras: T0 (1950–1995), T1 (1996–2010),
T2 (2011–2017), and T3 (2018 onward).

The first calendar API call seeds the database. You can also refresh it from the
project folder:

```powershell
python -m src.pipeline sync-calendar
```

The dashboard adds a **Race Calendar**, **Next Race Prediction**,
**Prediction Ledger**, and **Race Reports** view. The append-only P0 early and
P1 post-qualifying forecast ledger stores timestamped JSON files at
`predictions/ledger/` as well as SQLite rows. The upcoming-race workspace shows
the countdown and viewer-local session schedule, forecast stage, stop-count
ranking, strategy alternatives, scenario stress tests, circuit/history context,
input availability, and model provenance. Calendar changes are retained and
shown as alerts.
The database is local at `data/processed/race_registry.sqlite` by default; set
`RACE_DB_PATH` to choose another location.

P0 generation is permitted before the scheduled start; P1 generation is
intentionally time-gated and requires a scheduled race with known
race and qualifying times, starts no earlier than 60 minutes after qualifying
ends, refuses to run after lights out, and requires at least two earlier races
of real FastF1 or OpenF1 data. It trains only on earlier race dates, uses the latest
earlier race for chronological validation, and records the training cutoff,
model/source hashes, assumptions, and strategy ranking. A same-circuit prior
race is required to estimate race distance and typical conditions. Missing
historical data from both providers, or incomplete timing metadata, produces
an explicit error; demo laps are never accepted as training data or written
into the prediction ledger. OpenF1 race timing is enriched from its timestamped track-status, interval, and
position feeds. The dashboard aligns those signals to lap start times and shows
track status, gap to the car ahead, running position, and lap-to-lap position
change in Race Explorer. Coverage depends on the selected session; missing
published values remain blank and are not inferred as green flag, clear air, or
zero position change.

The current forecast model does not ingest practice long runs, qualifying-grid
positions, race-weekend weather forecasts, or tyre nominations. The P1 label
means the model was run after qualifying; it does not imply qualifying data
changed the model inputs. Rain is not modelled. Safety-car and cooler-track
controls are sensitivity tests, not claims about actual conditions. The
what-if tool accepts a user strategy and compares its simulated lap-time
profile against the current model pick. Pre-race picks use a random identifier
stored in the browser and are not authenticated accounts. Browser notifications
are best-effort while the dashboard remains open; email and Telegram delivery
require an external provider and are not configured.

Run lifecycle commands:

```powershell
python -m src.pipeline predict --race-id <race_id>
python -m src.pipeline predict-due
python -m src.pipeline ingest --race-id <race_id>
python -m src.pipeline ingest-due
python -m src.pipeline evaluate --race-id <race_id>
python -m src.pipeline evaluate-due
python -m src.pipeline backtest --start-year 2018 --end-year 2026
```

`race_id` is assigned by the registry (for example `2026_19`). `predict-due`
processes only events whose qualifying session ended at least an hour earlier
and whose race has not started. `ingest-due` waits at least five hours after
race start so incomplete sessions are not scored. Actual-result ingestion
currently uses FastF1 and requires F1 credentials. The walk-forward backtest
keeps backtest results separate from the public P1 ledger. Backtesting the full
modeling era can take substantial time and requires real, cached timing data;
OpenF1 historical data can be used for model training without FastF1 credentials.

Three GitHub Actions workflows are included for daily calendar sync, hourly
pre-race checks, and hourly post-race ingestion/evaluation retries. Enable them
after placing this project in its own GitHub repository. Configure the
repository Actions secrets `F1_USERNAME` and `F1_PASSWORD` if you want FastF1
tried first or need the current FastF1-based actual-result ingestion. OpenF1
historical predictions do not require these secrets. The prediction and
scorecard workflows publish generated JSON files as commits so timestamps can
be checked independently. Repository write permissions and an unprotected
automation branch are required for those scheduled commits.

The calendar, ledger, and historical OpenF1-backed P1 calls can work without
F1 credentials; post-race scoring still requires FastF1 credentials. The calendar source may differ from a static
season outline as race dates, venues, rounds, and cancellations change; the
dashboard reports the retrieved source and last-sync timestamp.

## Race-factor simulation

The lap-time model now includes team, tyre compound and age, fuel-load proxy, track
and air temperature, traffic pressure inferred from timing gaps, and position gains
as an overtaking signal. The simulator lets users model weather drift, traffic and
overtake rates, safety-car chance/timing/duration and pit-loss savings, and available
sets per compound. Monte Carlo outcomes show expected time, P10–P90 range, and
strategy win frequency. Traffic/overtaking effects are learned associations, not a
track-position or passing-physics engine; safety-car probabilities and tire sets
are scenario assumptions, not predictions of race control or the official allocation.

The app remains an educational, dry-race strategy simulator. It does not simulate
full live race control, red-flag restarts, detailed tyre degradation physics, or
reliable cross-season team performance. Synthetic fallback data is only for
exercising the application. Recommendations are not race-engineering advice.
