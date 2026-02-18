#!/usr/bin/env python3
"""CSS Grabber web application."""

from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_file

import css_grabber

app = Flask(__name__, static_folder=".", static_url_path="")


def make_slug(url: str) -> str:
    host = (urlparse(url).hostname or "report").lower()
    host = host.replace("www.", "")
    safe = "".join(ch if ch.isalnum() else "-" for ch in host).strip("-")
    return safe or "report"


def is_valid_target(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    return p.scheme in {"http", "https"} and bool(p.netloc)


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/api/report")
def generate_report():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or request.form.get("url") or "").strip()

    if not is_valid_target(url):
        return jsonify({"error": "Enter a valid http(s) URL."}), 400

    slug = make_slug(url)

    try:
        with tempfile.TemporaryDirectory(prefix="css-grabber-") as td:
            output = Path(td) / f"{slug}-report.html"
            css_grabber.run(url, output)
            return send_file(
                output,
                mimetype="text/html; charset=utf-8",
                as_attachment=True,
                download_name=f"{slug}-report.html",
            )
    except Exception as exc:
        return jsonify({"error": f"Report generation failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
