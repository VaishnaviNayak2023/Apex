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

## Deploy to Railway

The app is a monorepo with two independently deployed Railway services. Deploy
the same repository twice, using the project root as the build context:

Before connecting a repository, make sure it contains only this project. In the
current workstation, the Git root is `C:\Users\prash` and its `origin` points
to a different application; do not connect or push that home-directory
repository to Railway. Create/use a dedicated repository for this project,
with `package.json` and both Dockerfiles at its root.

1. Create an **API** service and set its Dockerfile path to `Dockerfile.api`.
   Set `PORT=8000` and the health-check path to `/api/health`.
2. Attach a Railway volume to the API service at `/app/data`. Keep this service
   at one replica: SQLite and its WAL files require a single writer and the
   volume is not shared across replicas.
3. Set these API service variables:

   ```text
   RACE_STORAGE_DIR=/app/data
   RACE_DB_PATH=/app/data/processed/race_registry.sqlite
   RACE_MODEL_DIR=/app/data/models
   RACE_LEDGER_DIR=/app/data/predictions/ledger
   ALLOW_DEMO_DATA=false
   F1_USERNAME=<set as a Railway secret if using FastF1>
   F1_PASSWORD=<set as a Railway secret if using FastF1>
   ```

   FastF1 credentials are optional; public OpenF1 historical data remains
   available without them. Never put credentials in the repository, Docker
   image, or frontend service variables.
4. Create a **web** service from the same repository and set its Dockerfile
   path to `Dockerfile.web`. Set `RACE_API_URL` to
   `http://api.railway.internal:8000` (the API service must be named `api` and
   listen on port 8000). Generate a Railway public domain for this service.
5. Deploy the API first, confirm its `/api/health` check passes, then deploy
   the web service. Open the public web domain and verify the calendar and
   historical race data load.

Railway build configuration must use the repository/project root as the build
context; the Dockerfiles are at that root. Only the API needs the persistent
volume. Its database, downloaded timing cache, forecast model artifacts, and
prediction ledger are configured under `/app/data`, so they survive API
releases. Back up this volume separately; it is not a substitute for backups.
The web container receives no F1 credentials.

The API and frontend Dockerfiles can also be built locally with Docker. The
frontend build uses Next.js standalone output; the API listens on Railway's
`PORT` and binds to `0.0.0.0`.

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
