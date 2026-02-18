# CSS Grabber

CSS Grabber can run as:
- A CLI tool for one-off reports
- A web app with a URL form + downloadable HTML report

## CLI Usage

```bash
python3 css_grabber.py https://example.com -o report.html
```

## Web App (Local)

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Start server:

```bash
python3 app.py
```

3. Open:

`http://localhost:8000`

The form posts to `/api/report` and downloads the generated report HTML.

## Deploy

This repo includes:
- `requirements.txt`
- `Procfile` (`gunicorn ... app:app`)

That is enough for simple deployment on platforms like Render/Railway/Fly/Heroku-style runtimes.

## Privacy/Data Flow

- The web app processes submitted URLs server-side.
- It fetches target page/CSS assets to build the report.
- Generated report files are created in a temporary directory per request.
