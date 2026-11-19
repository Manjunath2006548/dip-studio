# DIP Studio — Histogram Equalization Workbench

Interactive histogram-equalization workbench created by **Manjunath Bhaskar Hebbar**.

**Live on Vercel:** https://dip-studio-gilt.vercel.app

It runs the complete histogram-equalization experiment described in the report — image loading, centre-cropping, grayscale conversion, histogram / PDF / CDF computation, the equalization mapping, CLAHE and contrast-stretching comparisons, and all quality metrics — and presents the results as figures and tables in a browser. It also simulates the midday, evening and night versions of the day photograph, so the four illumination conditions can be reproduced from a single upload.

## Quick start

```bat
cd dip-studio
python -m pip install -r requirements.txt
python app.py
```

Then open **http://localhost:8000** — you land on the front page. Create an account (or log in), then use the workbench at **http://localhost:8000/app**.

Registration asks for a **username**, an **email address** and a password. The email format is validated in the browser and again on the server (for example `name@example.com`), and duplicate addresses are rejected — no mail is sent and no OTP is used. Login stays username + password. Passwords must be **12+ characters** and include an uppercase letter, a lowercase letter, a number and a special character; common phrases are rejected. The registration form shows a live strength meter. Accounts are stored locally in `data/users.db` with PBKDF2-SHA256 hashes and a signed, HTTP-only session cookie.

`run.bat` does the same on Windows: it installs the dependencies and starts the server.

## What you can do

- **Use the built-in sample** — the report's forest-stream day photograph is preloaded and analysed automatically on the first page load.
- **Upload your own photograph** — in *Simulate 4 illuminations* mode its lighting condition is detected automatically (day / midday / evening / night). The photo is analysed as the detected condition, and a day-like base is recovered from it to generate the other three with the same exposure gains, tints, glare/lamp terms and sensor noise as the report. The detected condition and its confidence are shown in the sidebar.
- **Upload four real photographs** — in *Use 4 photos as they are* mode each upload is assigned to day / midday / evening / night and processed directly.
- **Tune the experiment** — working size, CLAHE clip limit and tile grid, contrast-stretch percentiles, and the simulation exposure gains.
- **Inspect the results** — per-condition 2 × 3 panels, before/after slider, technique comparison grids and charts, and five result tables.
- **Export** — metrics as CSV or JSON, and figures as JPEG (click any figure).

## Reproduced results

For every illumination condition the workbench produces:

1. Original colour image, grayscale image and equalized image.
2. Original histogram with its CDF.
3. Transformation curve `s = T(r)` from Eq. (5).
4. Equalized histogram with its CDF.
5. Mean, contrast σ, Michelson contrast, dynamic range, occupied grey levels, entropy, both SNRs, sharpness, MSE, PSNR and AMBE.
6. A direct cross-check of the user-defined equalization against `cv2.equalizeHist` (expected maximum pixel difference: **0**).
7. Comparisons with CLAHE (clip limit 2.0, 8 × 8 tiles) and 1–99 % linear contrast stretching.

## Project layout

```
dip-studio/
├── app.py            FastAPI server and REST endpoints
├── dip_core.py       user-defined preprocessing, histogram, PDF/CDF, HE, metrics
├── simulation.py     day → midday / evening / night illumination simulation
├── plotting.py       Matplotlib panels, overview grid and comparison charts
├── samples/day.jpg   the report's real day photograph
├── static/
│   ├── landing.html   front page
│   ├── auth.html      registration / login
│   ├── site.css       landing + account styling
│   ├── index.html     workbench interface
│   └── images/        vector hero diagram (no photographs)
├── data/              users.db and session key (created on first run)
├── requirements.txt
├── run.bat
└── README.md
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Photo-free front page with founder credit and account links |
| `GET` | `/register`, `/login` | Registration and login pages |
| `GET` | `/app` | Workbench (requires a session) |
| `POST` | `/api/register`, `/api/login`, `/api/logout` | Account creation, sign-in, sign-out |
| `GET` | `/api/me` | Current signed-in username |
| `GET` | `/api/health` | Library versions and available conditions |
| `POST` | `/api/run` | Multipart form: `files`, `mode` (`simulate` / `direct`), `labels`, `params` (JSON) |
| `GET` | `/api/figure/{run_id}/{kind}[/{cond}]` | Renders one figure on first request, then serves it from cache (`kind` = `panel`, `overview`, `metrics`, `techniques`) |

Example request:

```bash
curl -X POST http://localhost:8000/api/run \
  -F "mode=simulate" \
  -F "params={\"width\":640,\"height\":480,\"clip_limit\":2.0,\"tile\":8}" \
  -F "files=@photo.jpg"
```

With no files the server uses `samples/day.jpg`.

## Implementation notes

- The histogram, PDF, CDF, equalization mapping, percentile stretch, Shannon entropy and Immerkaer noise estimate are all user-defined NumPy functions; OpenCV is used for I/O, resizing, CLAHE and as the independent equalization reference.
- Inputs are centre-cropped to the working aspect ratio and resized with `INTER_AREA` before any measurement, so all four conditions are directly comparable.
- Simulation works in linear light (γ = 2.2) with exposure gains 2.6× (midday), 0.20× (evening) and 0.012× (night), a warm evening tint, a cool night tint, graded sun glare, a localised warm lamp pool, and signal-dependent sensor noise that grows as the scene darkens.
- The dark end of the night mapping magnifies sensor noise; that limitation of global HE is visible in the equalized night image and discussed in the report.

## Performance

The interface stays responsive because expensive work is deferred and reused:

- **Embedded figures** — `/api/run` analyses the four conditions and returns every figure as an inline JPEG data URL. Nothing depends on a separate server round trip, so the *Techniques* grid (Figure 7 equivalent), comparison charts and per-condition panels render reliably on stateless hosts such as Vercel, where a later request could otherwise reach a different function instance. The `/api/figure/...` endpoint remains as a lightweight fallback and caches the same images server-side for local runs.
- **Run cache** — results are keyed by a SHA-256 of the parameters and image bytes; a repeated run returns in a few milliseconds. The most recent four runs are kept in memory.
- **Cheap rendering** — figures are JPEG, use `interpolation="none"` for photographs, draw histograms with one `stairs` artist instead of 256 bars, and down-sample grid thumbnails to 320 px before plotting.
- **Light payloads** — the before/after slider uses 512 px JPEG thumbnails instead of full-size images, so `/api/run` returns about 0.6 MB instead of several MB of base64.

The KPI cards show the pipeline time and the status line reports cache hits separately.

## Requirements

Python 3.12+, FastAPI, Uvicorn, OpenCV, NumPy, Matplotlib, python-multipart and Mangum (Vercel only). The interface has no external JavaScript or font dependencies and works offline once the server is running.

## Deploying to Vercel

The project is preconfigured for Vercel's Python runtime:

1. Install the CLI and log in: `npm i -g vercel` then `vercel login`.
2. From this directory run `vercel` (preview) or `vercel --prod` (production).

`vercel.json` routes every non-static path to `api/index.py`, which wraps the
same FastAPI app with Mangum. Static files under `static/` are served directly
by Vercel. The bundled `samples/day.jpg`, `static/` and `data/` are included in
the function via `includeFiles`.

Vercel serverless functions have a read-only source tree, so on Vercel the app
copies `data/users.db` into `/tmp/dip-studio` on a cold start and uses
`DIP_SESSION_SECRET` (or the bundled `data/session.key`) to sign cookies. That
`/tmp` database is discarded when an instance restarts, so registration also
sets a signed, HTTP-only **`dip_account` passport cookie** containing the
username, email and password hash. On the next login from that same browser the
app verifies the passport, rebuilds the lost database row and signs in — no
external database is needed. Logging out clears only the short-lived session,
not the passport. Accounts created before this fallback need one fresh
registration; the bundled `manjunath` account is always available. For accounts
that must work from a different browser or device, connect an external database
(e.g. Vercel Postgres) and point `_db()` at it.

Figures are embedded directly in the `/api/run` response, so the technique
grid and charts render even if later requests reach a different serverless
instance.
