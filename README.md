# CSS Grabber

Find what font, colors, and logo any website uses.

CSS Grabber is a free web tool and Python engine that analyzes a public URL and generates a one-page brand style report with:
- Logo detection + source URL
- Typography hierarchy (H1-H5 + body) with live samples
- Core and role-based brand colors in hex

Live app: [cssgrabber.com](https://cssgrabber.com)  

## Why teams use CSS Grabber
- UI designers: fast visual teardown without digging through DevTools
- Developers: align implementation to live site styles quickly
- Agencies/freelancers: accelerate discovery and kickoff audits
- Marketing/brand teams: competitor style reconnaissance

## How it works
1. Submit a public URL.
2. CSS Grabber fetches HTML/CSS/assets server-side.
3. Engine parses selectors, declarations, inline styles, and metadata.
4. Report renders as clean HTML ready to review, print, or save as PDF.

## Project structure
- `css_grabber.py` - extraction engine and report renderer
- `app.py` - Flask web app (`/` and `/api/report`)
- `index.html` - SEO landing + form UI
- `output.html` - hosted report viewer
- `terms.html` - terms and privacy page
- `Procfile` - Gunicorn startup for hosting platforms

## Local development

### Requirements
- Python 3.10+

### Install
```bash
python3 -m pip install -r requirements.txt
```

### Run web app
```bash
python3 app.py
```
Open [http://localhost:8000](http://localhost:8000).

### Run CLI
```bash
python3 css_grabber.py https://example.com -o report.html
```

## Deployment
This repo is configured for simple deployment on Render/Railway/Fly/Heroku-style runtimes:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- `Procfile` included

## Privacy and data flow
- URL analysis is processed server-side.
- Target HTML/CSS/assets are fetched to compute report tokens.
- Reports are generated in temporary request-scoped files.
- Terms and Privacy: [cssgrabber.com/terms.html](https://cssgrabber.com/terms.html)

## Known limitations
- Bot-protected sites may return challenge pages or partial extraction.
- Font rendering can depend on remote font host policies and CORS behavior.
- Dynamic/JS-hydrated states may require page-specific sampling.

## Reporting issues
Use GitHub Issues for extraction mismatches:
- `Bug report: extraction mismatch`
- `Site request: analyze this URL`

Please include:
- Target URL
- What was expected vs what was extracted
- Screenshot when possible

## Roadmap themes
- Extraction quality benchmarking and regression fixtures
- Better anti-bot classification and recovery messaging
- Optional API mode and batch processing

## License
No license file has been added yet. If you plan open contribution, add a license (MIT is common).
