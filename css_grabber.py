#!/usr/bin/env python3
"""CSS Grabber: extract logo, typography, and brand colors from a URL."""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import ssl
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT = 20
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
TARGET_ELEMENTS = ["body", "h1", "h2", "h3", "h4", "h5", "a", "button"]
FONT_PROPS = ["font-family", "font-size", "font-weight", "line-height", "letter-spacing"]
COLOR_PROPS = ["color", "background-color", "border-color"]
ALL_PROPS = FONT_PROPS + COLOR_PROPS


@dataclass
class CSSRule:
    selector: str
    declarations: Dict[str, Tuple[str, bool]]
    order: int


@dataclass
class LogoCandidate:
    src: str
    score: int
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class BrandRoleResult:
    key: str
    label: str
    value: str
    evidence: str
    status: str


@dataclass
class FontFaceEntry:
    family: str
    src: str
    style: str
    weight: str
    display: str


def is_ignored_font_family(name: str) -> bool:
    n = (name or "").strip().lower()
    return any(token in n for token in ("dashicons", "fontawesome", "material icons", "icon"))


class SimpleHTMLIndex(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inline_styles: Dict[str, Dict[str, str]] = {}
        self.first_element_attrs: Dict[str, Dict[str, object]] = {}
        self.stylesheets: List[str] = []
        self.style_blocks: List[str] = []
        self.title_text = ""
        self.site_name = ""
        self.logo_candidates: List[LogoCandidate] = []
        self.meta_logo: Optional[str] = None
        self.site_icons: List[str] = []

        self._in_style = False
        self._style_buf: List[str] = []
        self._in_title = False
        self._title_buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}

        if tag in TARGET_ELEMENTS and "style" in attr_map and tag not in self.inline_styles:
            self.inline_styles[tag] = parse_declarations(attr_map["style"])

        if tag in TARGET_ELEMENTS and tag not in self.first_element_attrs:
            classes = {c for c in attr_map.get("class", "").split() if c}
            self.first_element_attrs[tag] = {"id": attr_map.get("id", ""), "classes": classes}

        if tag == "link":
            rel = attr_map.get("rel", "").lower()
            href = attr_map.get("href", "")
            if "stylesheet" in rel and href:
                self.stylesheets.append(href)
            if "icon" in rel and href:
                self.site_icons.append(href)

        if tag == "img":
            src = pick_img_src(attr_map)
            if src:
                self.logo_candidates.append(
                    LogoCandidate(
                        src=src,
                        score=score_logo_candidate(attr_map),
                        width=parse_px_int(attr_map.get("width", "")),
                        height=parse_px_int(attr_map.get("height", "")),
                    )
                )

        if tag == "meta":
            prop = (attr_map.get("property", "") or attr_map.get("name", "")).lower()
            content = attr_map.get("content", "")
            if content and prop in {"og:logo", "twitter:logo", "logo"}:
                self.meta_logo = content
            if content and prop in {"og:site_name", "application-name", "twitter:site"} and not self.site_name:
                self.site_name = content.strip().lstrip("@")

        if tag == "style":
            self._in_style = True
            self._style_buf = []
        elif tag == "title":
            self._in_title = True
            self._title_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "style" and self._in_style:
            self._in_style = False
            css = "".join(self._style_buf).strip()
            if css:
                self.style_blocks.append(css)
        elif tag == "title" and self._in_title:
            self._in_title = False
            title = " ".join("".join(self._title_buf).split())
            if title:
                self.title_text = title

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self._style_buf.append(data)
        elif self._in_title:
            self._title_buf.append(data)


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as res:
            charset = res.headers.get_content_charset() or "utf-8"
            body = res.read()
    except (ssl.SSLCertVerificationError, URLError) as exc:
        should_retry = isinstance(exc, ssl.SSLCertVerificationError)
        if isinstance(exc, URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
            should_retry = True
        if not should_retry:
            raise
        with urlopen(req, timeout=timeout, context=ssl._create_unverified_context()) as res:
            charset = res.headers.get_content_charset() or "utf-8"
            body = res.read()
    return body.decode(charset, errors="replace")


def fetch_bytes(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as res:
            return res.read()
    except (ssl.SSLCertVerificationError, URLError) as exc:
        should_retry = isinstance(exc, ssl.SSLCertVerificationError)
        if isinstance(exc, URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
            should_retry = True
        if not should_retry:
            raise
        with urlopen(req, timeout=timeout, context=ssl._create_unverified_context()) as res:
            return res.read()


def infer_image_mime(url: str, blob: bytes) -> str:
    lower = url.lower()
    path = urlparse(url).path.lower()
    if path.endswith(".svg") or blob.lstrip().startswith(b"<svg"):
        return "image/svg+xml"
    if path.endswith(".png") or blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if path.endswith(".webp") or blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    if path.endswith(".gif") or blob.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if path.endswith(".ico") or path.endswith(".cur"):
        return "image/x-icon"
    if path.endswith(".jpg") or path.endswith(".jpeg") or blob.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if ".svg" in lower:
        return "image/svg+xml"
    if ".png" in lower:
        return "image/png"
    if ".webp" in lower:
        return "image/webp"
    if ".gif" in lower:
        return "image/gif"
    if ".ico" in lower:
        return "image/x-icon"
    if ".jpg" in lower or ".jpeg" in lower:
        return "image/jpeg"
    return "image/png"


def image_data_uri(url: str) -> Optional[str]:
    try:
        blob = fetch_bytes(url)
    except Exception:
        return None
    if not blob:
        return None
    mime = infer_image_mime(url, blob)
    return f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}"


def strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def parse_declarations(block: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in block.split(";"):
        if ":" not in part:
            continue
        prop, val = part.split(":", 1)
        prop = prop.strip().lower()
        val = val.strip()
        if prop and val:
            out[prop] = val
    return out


def parse_rule_declarations(block: str) -> Dict[str, Tuple[str, bool]]:
    out: Dict[str, Tuple[str, bool]] = {}
    for part in block.split(";"):
        if ":" not in part:
            continue
        prop, val = part.split(":", 1)
        prop = prop.strip().lower()
        val = val.strip()
        if not prop or not val:
            continue
        important = False
        if "!important" in val:
            important = True
            val = re.sub(r"\s*!important\s*", "", val, flags=re.I).strip()
        out[prop] = (val, important)
    return out


def split_rules(css: str) -> List[Tuple[str, str]]:
    css = strip_comments(css)
    pairs: List[Tuple[str, str]] = []
    i = 0
    n = len(css)
    while i < n:
        while i < n and css[i].isspace():
            i += 1
        if i >= n:
            break

        head_start = i
        while i < n and css[i] != "{":
            i += 1
        if i >= n:
            break

        head = css[head_start:i].strip()
        i += 1
        depth = 1
        body_start = i
        while i < n and depth > 0:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1

        body = css[body_start : i - 1]
        if head:
            pairs.append((head, body))
    return pairs


def expand_css_rules(css_text: str, start_order: int = 0) -> Tuple[List[CSSRule], int]:
    rules: List[CSSRule] = []
    order = start_order
    for head, body in split_rules(css_text):
        if head.startswith("@"):
            nested, order = expand_css_rules(body, order)
            rules.extend(nested)
            continue
        decls = parse_rule_declarations(body)
        if not decls:
            continue
        for selector in [s.strip() for s in head.split(",") if s.strip()]:
            rules.append(CSSRule(selector=selector, declarations=decls, order=order))
            order += 1
    return rules, order


def remove_pseudo(selector: str) -> str:
    return re.sub(r"::?[a-zA-Z-]+(?:\([^)]*\))?", "", selector).strip()


def calc_specificity(selector: str) -> Tuple[int, int, int]:
    s = remove_pseudo(selector)
    ids = len(re.findall(r"#[a-zA-Z0-9_-]+", s))
    classes = len(re.findall(r"\.[a-zA-Z0-9_-]+", s)) + len(re.findall(r"\[[^\]]+\]", s))
    types = len(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", re.sub(r"[#.][a-zA-Z0-9_-]+", "", s)))
    return (ids, classes, types)


def selector_targets_element(selector: str, element: str, sample: Optional[Dict[str, object]]) -> bool:
    s = remove_pseudo(selector)
    if not s:
        return False

    if element == "body" and s in ("html", ":root", "*", "body"):
        return True

    parts = re.split(r"\s+|>|\+|~", s)
    tail = parts[-1].strip() if parts else ""
    if not tail:
        return False

    if tail in ("*", "html", ":root") and element == "body" and len(parts) == 1:
        return True

    sample_id = str((sample or {}).get("id", "") or "")
    sample_classes: Set[str] = set((sample or {}).get("classes", set()) or set())

    ids = re.findall(r"#([a-zA-Z0-9_-]+)", tail)
    classes = re.findall(r"\.([a-zA-Z0-9_-]+)", tail)
    if ids and (not sample_id or any(found != sample_id for found in ids)):
        return False
    if classes and (not sample_classes or any(found not in sample_classes for found in classes)):
        return False

    m = re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*", tail)
    if not m:
        return bool(ids or classes)
    return m.group(0).lower() == element


def selector_targets_element_loose(selector: str, element: str) -> bool:
    s = remove_pseudo(selector).lower()
    if not s:
        return False
    if element == "body":
        if s in ("body", "html", ":root", "*"):
            return True
        return "body" in s
    parts = re.split(r"\s+|>|\+|~", s)
    tail = parts[-1].strip() if parts else ""
    if not tail:
        return False
    if tail == element:
        return True
    if tail.startswith(element + ".") or tail.startswith(element + "#"):
        return True
    return False


def resolve_vars(value: str, var_map: Dict[str, str], depth: int = 0) -> str:
    if depth > 6:
        return value
    pattern = re.compile(r"var\((--[a-zA-Z0-9_-]+)(?:\s*,\s*([^\)]+))?\)")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        fallback = (match.group(2) or "").strip()
        if key in var_map:
            return resolve_vars(var_map[key], var_map, depth + 1)
        return fallback

    return pattern.sub(repl, value)


def build_var_map(rules: List[CSSRule]) -> Dict[str, str]:
    var_map: Dict[str, str] = {}
    for rule in rules:
        for prop, (val, _) in rule.declarations.items():
            if prop.startswith("--"):
                var_map[prop] = val
    return var_map


def compute_styles(
    rules: List[CSSRule],
    inline_styles: Dict[str, Dict[str, str]],
    var_map: Dict[str, str],
    element_samples: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, str]]:
    computed: Dict[str, Dict[str, str]] = {e: {} for e in TARGET_ELEMENTS}
    weight: Dict[Tuple[str, str], Tuple[int, Tuple[int, int, int], int]] = {}

    for element in TARGET_ELEMENTS:
        sample = element_samples.get(element)
        for rule in rules:
            if not selector_targets_element(rule.selector, element, sample):
                continue
            spec = calc_specificity(rule.selector)
            for prop, (val, important) in rule.declarations.items():
                if prop not in ALL_PROPS:
                    continue
                key = (element, prop)
                score = (1 if important else 0, spec, rule.order)
                if key not in weight or score >= weight[key]:
                    weight[key] = score
                    computed[element][prop] = resolve_vars(val, var_map)

    for element, styles in inline_styles.items():
        if element not in computed:
            continue
        for prop, val in styles.items():
            if prop in ALL_PROPS:
                computed[element][prop] = resolve_vars(val, var_map)

    fallback_weight: Dict[Tuple[str, str], Tuple[int, Tuple[int, int, int], int]] = {}
    for element in TARGET_ELEMENTS:
        for rule in rules:
            if not selector_targets_element_loose(rule.selector, element):
                continue
            spec = calc_specificity(rule.selector)
            for prop, (val, important) in rule.declarations.items():
                if prop not in ALL_PROPS or prop in computed[element]:
                    continue
                key = (element, prop)
                score = (1 if important else 0, spec, rule.order)
                if key not in fallback_weight or score >= fallback_weight[key]:
                    fallback_weight[key] = score
                    computed[element][prop] = resolve_vars(val, var_map)

    inherited = ["font-family", "font-size", "font-weight", "line-height", "letter-spacing", "color"]
    body_base = computed.get("body", {})
    for element in TARGET_ELEMENTS:
        if element == "body":
            continue
        for prop in inherited:
            if prop not in computed[element] and prop in body_base:
                computed[element][prop] = body_base[prop]

    return computed


def parse_px_int(value: str) -> Optional[int]:
    m = re.search(r"\d+", value or "")
    return int(m.group(0)) if m else None


def pick_img_src(attrs: Dict[str, str]) -> str:
    for key in ("src", "data-src", "data-lazy-src", "data-original", "data-orig-file"):
        val = attrs.get(key, "").strip()
        if val:
            return val
    for key in ("srcset", "data-lazy-srcset"):
        srcset = attrs.get(key, "").strip()
        if srcset:
            return srcset.split(",", 1)[0].strip().split(" ", 1)[0].strip()
    return ""


def score_logo_candidate(attrs: Dict[str, str]) -> int:
    src = pick_img_src(attrs)
    hay = " ".join([attrs.get("class", ""), attrs.get("id", ""), attrs.get("alt", ""), src]).lower()
    score = 0
    for token in ("site-logo", "custom-logo", "header-logo", "brand-logo"):
        if token in hay:
            score += 8
    for token in ("logo", "brand", "header"):
        if token in hay:
            score += 3
    for token in ("icon", "avatar", "sprite", "hero", "ad-"):
        if token in hay:
            score -= 4
    if src.startswith("data:image"):
        score -= 20

    width = parse_px_int(attrs.get("width", ""))
    height = parse_px_int(attrs.get("height", ""))
    if width and height:
        area = width * height
        if 1200 <= area <= 140000:
            score += 2
        if area > 400000:
            score -= 2
    return score


def pick_logo_url(page_url: str, parser: SimpleHTMLIndex, schema_logo_url: Optional[str] = None) -> Optional[str]:
    if not parser.logo_candidates:
        if schema_logo_url:
            return urljoin(page_url, schema_logo_url)
        return urljoin(page_url, parser.meta_logo) if parser.meta_logo else None

    host = (urlparse(page_url).hostname or "").lower().replace("www.", "")
    tokens = [t for t in re.split(r"[^a-z0-9]+", host) if t and t not in {"com", "net", "org"}]

    best_by_src: Dict[str, int] = {}
    for c in parser.logo_candidates:
        resolved = urljoin(page_url, c.src)
        s = c.score
        parsed = urlparse(resolved)
        pathish = (parsed.path + "?" + parsed.query).lower()
        lower = pathish
        if parsed.path.endswith("/_next/image") and parsed.query:
            q = parse_qs(parsed.query)
            img = unquote((q.get("url") or [""])[0]).lower()
            if img:
                lower = img

        if tokens and all(t in lower for t in tokens):
            s += 16
        elif tokens and any(t in lower for t in tokens):
            s += 6

        if "logo" in lower:
            s += 2
        if "banner" in lower:
            s += 2
        if "grey" in lower or "grayscale" in lower:
            s -= 5

        if c.width and c.height:
            ratio = c.width / c.height if c.height else 0.0
            if 2.0 <= ratio <= 8.0:
                s += 5
            if ratio < 1.0:
                s -= 2

        if resolved not in best_by_src or s > best_by_src[resolved]:
            best_by_src[resolved] = s

    if best_by_src:
        if schema_logo_url:
            resolved_schema = urljoin(page_url, schema_logo_url)
            # Schema logos can be stale in CMS settings, so do not auto-win.
            # Keep a modest baseline score and let stronger in-page header logos beat it.
            best_by_src[resolved_schema] = max(best_by_src.get(resolved_schema, -999), 12)
        best_url, best_score = sorted(best_by_src.items(), key=lambda item: item[1], reverse=True)[0]
        # Avoid picking arbitrary content photos when no logo-like evidence exists.
        if best_score >= 10:
            return best_url

    if schema_logo_url:
        return urljoin(page_url, schema_logo_url)

    if parser.meta_logo:
        return urljoin(page_url, parser.meta_logo)

    if parser.site_icons:
        # Prefer non-SVG icon first for widest browser compatibility.
        for icon in parser.site_icons:
            if not icon.lower().endswith(".svg"):
                return urljoin(page_url, icon)
        return urljoin(page_url, parser.site_icons[0])

    return None


def extract_schema_logo_url(html_text: str, page_url: str) -> Optional[str]:
    script_blocks: List[str] = []
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", html_text, flags=re.I | re.S):
        attrs = (m.group(1) or "").lower()
        if "application/ld+json" not in attrs:
            continue
        script_blocks.append(html.unescape((m.group(2) or "").strip()))
    if not script_blocks:
        return None

    def collect_image_ids(node: object, ids: Dict[str, str]) -> None:
        if isinstance(node, dict):
            node_id = node.get("@id")
            node_type = node.get("@type")
            node_url = node.get("url") or node.get("contentUrl")
            if isinstance(node_id, str) and isinstance(node_url, str):
                if (
                    (isinstance(node_type, str) and "imageobject" in node_type.lower())
                    or (isinstance(node_type, list) and any("imageobject" in str(t).lower() for t in node_type))
                    or "/schema/logo/image" in node_id
                ):
                    ids[node_id] = node_url
            for value in node.values():
                collect_image_ids(value, ids)
        elif isinstance(node, list):
            for item in node:
                collect_image_ids(item, ids)

    def walk(node: object, ids: Dict[str, str]) -> Optional[str]:
        if isinstance(node, dict):
            logo = node.get("logo")
            if isinstance(logo, str) and logo.strip():
                return logo.strip()
            if isinstance(logo, dict):
                candidate = logo.get("url") or logo.get("contentUrl") or logo.get("@id")
                if isinstance(candidate, str) and candidate.strip():
                    resolved = ids.get(candidate, candidate)
                    return resolved.strip()
            if isinstance(logo, list):
                for item in logo:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                    if isinstance(item, dict):
                        candidate = item.get("url") or item.get("contentUrl") or item.get("@id")
                        if isinstance(candidate, str) and candidate.strip():
                            resolved = ids.get(candidate, candidate)
                            return resolved.strip()
            for value in node.values():
                found = walk(value, ids)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = walk(item, ids)
                if found:
                    return found
        return None

    for raw in script_blocks:
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        id_map: Dict[str, str] = {}
        collect_image_ids(data, id_map)
        found = walk(data, id_map)
        if found:
            return urljoin(page_url, found)
    return None


def rewrite_css_urls(value: str, base_url: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(1).strip().strip('"\'')
        if not raw or raw.startswith(("data:", "http://", "https://", "//", "#")):
            return f"url('{raw}')"
        return f"url('{urljoin(base_url, raw)}')"

    return re.sub(r"url\(([^)]+)\)", repl, value)


def extract_font_faces(css_text: str, base_url: str) -> List[str]:
    out: List[str] = []
    for head, body in split_rules(css_text):
        if head.strip().lower() != "@font-face":
            continue
        decls = parse_declarations(body)
        fam = decls.get("font-family")
        src = decls.get("src")
        if not fam or not src:
            continue
        src = rewrite_css_urls(src, base_url)
        face = [f"font-family: {fam};", f"src: {src};"]
        for p in ("font-style", "font-weight", "font-display", "unicode-range", "font-stretch"):
            if p in decls:
                face.append(f"{p}: {decls[p]};")
        out.append("@font-face { " + " ".join(face) + " }")
    return out


def extract_font_face_entries(css_text: str, base_url: str) -> List[FontFaceEntry]:
    entries: List[FontFaceEntry] = []
    for head, body in split_rules(css_text):
        if head.strip().lower() != "@font-face":
            continue
        decls = parse_declarations(body)
        family = clean_css_value(decls.get("font-family"))
        src = decls.get("src")
        if not family or not src:
            continue
        family_clean = family.strip().strip("\"'")
        src_rewritten = rewrite_css_urls(src, base_url)
        src_low = src_rewritten.lower()
        if is_ignored_font_family(family_clean):
            continue
        # Drop malformed/tricky inline application font URIs that can break style parsing.
        if "data:application/" in src_low:
            continue
        if src_rewritten.count("'") % 2 != 0 or src_rewritten.count('"') % 2 != 0:
            continue
        entries.append(
            FontFaceEntry(
                family=family_clean,
                src=src_rewritten,
                style=clean_css_value(decls.get("font-style")) or "normal",
                weight=clean_css_value(decls.get("font-weight")) or "400",
                display=clean_css_value(decls.get("font-display")) or "swap",
            )
        )
    return entries


def first_font_url(src_value: str) -> Optional[str]:
    urls = re.findall(r"url\(([^)]+)\)", src_value)
    for raw in urls:
        url = raw.strip().strip("\"'")
        if not url:
            continue
        if url.startswith("data:"):
            return url
        if ".woff2" in url or ".woff" in url or ".ttf" in url:
            return url
    return None


def inline_used_font_faces(entries: List[FontFaceEntry], used_families: Set[str]) -> str:
    by_family: Dict[str, List[FontFaceEntry]] = {}
    for e in entries:
        by_family.setdefault(e.family.lower(), []).append(e)

    blocks: List[str] = []
    for fam in used_families:
        fam_key = fam.lower().strip().strip("\"'")
        if is_ignored_font_family(fam_key):
            continue
        if fam_key not in by_family:
            continue
        count = 0
        for entry in by_family[fam_key]:
            font_url = first_font_url(entry.src)
            if not font_url or font_url.startswith("data:"):
                continue
            try:
                data = fetch_bytes(font_url)
            except Exception:
                continue
            fmt = "woff2" if ".woff2" in font_url else "woff"
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:font/{fmt};base64,{b64}"
            blocks.append(
                "@font-face { "
                f"font-family: '{entry.family}'; "
                f"src: url('{data_url}') format('{fmt}'); "
                f"font-style: {entry.style}; "
                f"font-weight: {entry.weight}; "
                f"font-display: {entry.display}; "
                "}"
            )
            count += 1
            if count >= 3:
                break
    return "\\n".join(blocks)


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def clean_css_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return re.sub(r"\s*!important\s*", "", value, flags=re.I).strip()


def color_to_hex(value: Optional[str]) -> Optional[str]:
    token = extract_color_token(value)
    if not token:
        return None
    t = token.strip().lower()
    named = {
        "black": "#000000",
        "white": "#ffffff",
        "gray": "#808080",
        "grey": "#808080",
        "silver": "#c0c0c0",
        "red": "#ff0000",
        "green": "#008000",
        "blue": "#0000ff",
        "navy": "#000080",
        "teal": "#008080",
        "maroon": "#800000",
        "purple": "#800080",
        "orange": "#ffa500",
        "yellow": "#ffff00",
        "aqua": "#00ffff",
        "fuchsia": "#ff00ff",
        "lime": "#00ff00",
        "olive": "#808000",
        "transparent": "#000000",
    }
    if t in named:
        return named[t]
    if t.startswith("#"):
        h = t[1:]
        if len(h) == 3:
            return "#" + "".join(ch * 2 for ch in h)
        if len(h) == 4:
            return "#" + "".join(ch * 2 for ch in h[:3])
        if len(h) == 6:
            return "#" + h
        if len(h) == 8:
            return "#" + h[:6]
        return None
    m = re.match(r"rgba?\(([^)]+)\)", t)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) < 3:
            return None
        rgb = []
        for p in parts[:3]:
            if p.endswith("%"):
                try:
                    v = round(float(p[:-1]) * 2.55)
                except ValueError:
                    return None
            else:
                try:
                    v = int(float(p))
                except ValueError:
                    return None
            v = max(0, min(255, v))
            rgb.append(v)
        return "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])
    return None


def primary_font_name(font_family: Optional[str]) -> str:
    ff = clean_css_value(font_family)
    if not ff:
        return "not found"
    token = ff.split(",", 1)[0].strip().strip("\"'")
    if not token:
        return "not found"
    # Next.js font optimization often emits internal family names like
    # __Ubuntu_eefc74 or __DM_Sans_Fallback_abc123.
    m = re.match(r"^__([A-Za-z0-9_]+?)_[A-Fa-f0-9]+$", token)
    if m:
        cleaned = m.group(1).replace("_Fallback", "").replace("_", " ").strip()
        if cleaned:
            return cleaned
    return token


def extract_color_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = clean_css_value(value)
    if not v:
        return None
    v = v.strip()
    if v.lower() in {"inherit", "initial", "unset", "none", "auto"}:
        return None

    hex_match = re.search(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b", v)
    if hex_match:
        return hex_match.group(0).lower()

    func_match = re.search(r"(?:rgb|rgba|hsl|hsla)\([^\)]+\)", v, flags=re.I)
    if func_match:
        return func_match.group(0).lower()

    named = re.search(
        r"\b(black|white|gray|grey|silver|red|green|blue|navy|teal|maroon|purple|orange|yellow|aqua|fuchsia|lime|olive|transparent|currentcolor)\b",
        v,
        flags=re.I,
    )
    if named:
        return named.group(1).lower()

    return None


def has_hover_state(selector: str) -> bool:
    s = selector.lower()
    return any(token in s for token in (":hover", ":focus", ":active", ":focus-visible"))


def is_link_selector(selector: str) -> bool:
    s = selector.lower()
    if re.search(r"(^|[\s>+~,(])a([\s>+~:#.\[,)]|$)", s):
        return True
    return "link" in s


def is_button_selector(selector: str) -> bool:
    s = selector.lower()
    return any(token in s for token in ("button", ".btn", "submit", "wp-block-button", "cta", "readmore", "formkit"))


def is_icon_selector(selector: str) -> bool:
    s = selector.lower()
    return any(token in s for token in ("svg", "icon", "social", "fa-", "menu-toggle", "rating-star"))


def is_heading_selector(selector: str) -> bool:
    s = selector.lower()
    return any(token in s for token in ("h1", "h2", "h3", "entry-title", "site-title", "widget-title"))


def is_muted_selector(selector: str) -> bool:
    s = selector.lower()
    return any(token in s for token in ("muted", "meta", "caption", "byline", "small", "subhead", "subtitle"))


def is_page_surface_selector(selector: str) -> bool:
    s = selector.lower()
    return any(token in s for token in ("body", "html", "site-container", "site-wrap", "site-inner", "#page"))


def is_section_surface_selector(selector: str) -> bool:
    s = selector.lower()
    return any(token in s for token in ("section", "content", "container", "main", "wrap", "hero"))


def is_card_surface_selector(selector: str) -> bool:
    s = selector.lower()
    return any(token in s for token in ("card", "widget", "post", "entry", "tile", "panel", "box"))


def is_noise_selector(selector: str) -> bool:
    s = selector.lower()
    return any(
        token in s
        for token in (
            "wprm-",
            "dpsp-",
            "tippy",
            "wp-lightbox",
            "pin-it",
            "admin-bar",
            "nutrition-label",
            "rating-star",
            "wpgdprc",
            "convertkit",
            "editor-styles-wrapper",
            "social-brand-colors",
            "social-show-brand",
            "shared-counts",
        )
    )


def selector_quality(selector: str) -> int:
    s = selector.lower()
    score = 0
    if any(token in s for token in ("site-header", "main-navigation", "primary-menu", "header-navigation", "site-main", "entry-content", "content-area", "site-footer")):
        score += 4
    if any(token in s for token in ("button", ".btn", "formkit", "subscribe", "recipe-index", "cta")):
        score += 3
    if any(token in s for token in ("widget-drawer", "mobile-navigation", "off-canvas", "sidebar")):
        score -= 2
    return score


def role_selector_bonus(role: str, selector: str) -> int:
    s = selector.strip().lower()
    if role == "link.default":
        if any(token in s for token in ("nav", "menu", "header")) and "a" in s:
            return 20
        if s == "a":
            return 6
        if s.startswith("a:") and "hover" not in s:
            return 12
    if role == "link.hover":
        if s in {"a:hover", "a:focus", "a:active", "a:focus-visible"}:
            return 20
        if s.startswith("a:"):
            return 12
    if role == "text.body" and s in {"body", "html", ":root"}:
        return 16
    if role == "text.heading" and s in {"h1", "h2", "h3"}:
        return 10
    if role.startswith("button.primary") and s in {"button", "input[type=submit]", "input[type=\"submit\"]"}:
        return 10
    return 0


def collect_brand_profile(rules: List[CSSRule], var_map: Dict[str, str], styles: Dict[str, Dict[str, str]]) -> List[BrandRoleResult]:
    role_labels = {
        "text.body": "Text Body",
        "text.heading": "Text Heading",
        "text.muted": "Text Muted",
        "link.default": "Link Default",
        "link.hover": "Link Hover",
        "button.primary.bg": "Button Primary Background",
        "button.primary.text": "Button Primary Text",
        "button.primary.border": "Button Primary Border",
        "button.primary.hover.bg": "Button Primary Hover Background",
        "surface.page": "Surface Page",
        "surface.section": "Surface Section",
        "surface.card": "Surface Card",
        "border.default": "Border Default",
        "icon.default": "Icon Default",
        "icon.accent": "Icon Accent",
    }

    candidates: Dict[str, List[Tuple[Tuple[int, Tuple[int, int, int], int, int], str, str]]] = {k: [] for k in role_labels}

    def add(role: str, color: str, selector: str, important: bool, spec: Tuple[int, int, int], order: int, bonus: int = 0) -> None:
        if role not in candidates:
            return
        if color == "currentcolor":
            return
        if role != "surface.card" and color == "transparent":
            return
        extra = role_selector_bonus(role, remove_pseudo(selector))
        score = (1 if important else 0, bonus + extra, spec, order)
        candidates[role].append((score, color, selector))

    for rule in rules:
        sel = rule.selector
        sel_clean = remove_pseudo(sel)
        if is_noise_selector(sel_clean):
            continue
        spec = calc_specificity(sel)
        quality = selector_quality(sel_clean)
        hover = has_hover_state(sel)
        is_link = is_link_selector(sel_clean)
        is_button = is_button_selector(sel_clean)
        is_icon = is_icon_selector(sel_clean)
        is_heading = is_heading_selector(sel_clean)

        for prop, (raw, important) in rule.declarations.items():
            value = resolve_vars(raw, var_map)
            color = extract_color_token(value)
            if not color:
                continue

            if prop == "color":
                if "body" in sel_clean.lower() or sel_clean.lower() in {"html", "body", ":root", "*"}:
                    add("text.body", color, sel, important, spec, rule.order, 3 + quality)
                if is_heading:
                    add("text.heading", color, sel, important, spec, rule.order, 3 + quality)
                if is_muted_selector(sel_clean):
                    add("text.muted", color, sel, important, spec, rule.order, 2 + quality)
                if is_link and not hover:
                    add("link.default", color, sel, important, spec, rule.order, 3 + quality)
                if is_link and hover:
                    add("link.hover", color, sel, important, spec, rule.order, 3 + quality)
                if is_button and not hover:
                    add("button.primary.text", color, sel, important, spec, rule.order, 3 + quality)
                if is_icon and not hover:
                    add("icon.default", color, sel, important, spec, rule.order, 2 + quality)
                if is_icon and hover:
                    add("icon.accent", color, sel, important, spec, rule.order, 2 + quality)

            if prop == "background-color":
                if is_button and not hover:
                    add("button.primary.bg", color, sel, important, spec, rule.order, 3 + quality)
                if is_button and hover:
                    add("button.primary.hover.bg", color, sel, important, spec, rule.order, 3 + quality)
                if is_page_surface_selector(sel_clean):
                    add("surface.page", color, sel, important, spec, rule.order, 4 + quality)
                if is_section_surface_selector(sel_clean):
                    add("surface.section", color, sel, important, spec, rule.order, 2 + quality)
                if is_card_surface_selector(sel_clean):
                    add("surface.card", color, sel, important, spec, rule.order, 2 + quality)

            if prop == "border-color":
                if is_button and not hover:
                    add("button.primary.border", color, sel, important, spec, rule.order, 2 + quality)
                add("border.default", color, sel, important, spec, rule.order, 1 + quality)

            if prop in {"fill", "stroke"} and is_icon:
                if hover:
                    add("icon.accent", color, sel, important, spec, rule.order, 3 + quality)
                else:
                    add("icon.default", color, sel, important, spec, rule.order, 3 + quality)

    resolved: Dict[str, BrandRoleResult] = {}

    def pick(role: str) -> Optional[BrandRoleResult]:
        if not candidates[role]:
            return None
        if role == "link.default":
            nav_context = [
                c
                for c in candidates[role]
                if any(token in remove_pseudo(c[2]).lower() for token in ("nav", "menu", "header")) and "a" in remove_pseudo(c[2]).lower()
            ]
            if nav_context:
                best = sorted(nav_context, key=lambda item: item[0], reverse=True)[0]
                return BrandRoleResult(role, role_labels[role], best[1], best[2], "found")
            exact = [c for c in candidates[role] if remove_pseudo(c[2]).strip().lower() == "a"]
            if exact:
                best = sorted(exact, key=lambda item: item[0], reverse=True)[0]
                return BrandRoleResult(role, role_labels[role], best[1], best[2], "found")
        if role == "link.hover":
            exact = [
                c
                for c in candidates[role]
                if remove_pseudo(c[2]).strip().lower() in {"a:hover", "a:focus", "a:active", "a:focus-visible"}
            ]
            if exact:
                best = sorted(exact, key=lambda item: item[0], reverse=True)[0]
                return BrandRoleResult(role, role_labels[role], best[1], best[2], "found")
        best = sorted(candidates[role], key=lambda item: item[0], reverse=True)[0]
        return BrandRoleResult(role, role_labels[role], best[1], best[2], "found")

    for role in role_labels:
        got = pick(role)
        if got:
            resolved[role] = got

    body_color = extract_color_token(styles.get("body", {}).get("color")) or "#000"
    heading_color = extract_color_token(styles.get("h1", {}).get("color") or styles.get("h2", {}).get("color")) or body_color
    link_color = extract_color_token(styles.get("a", {}).get("color")) or heading_color
    page_bg = extract_color_token(styles.get("body", {}).get("background-color")) or "#ffffff"
    border_default = extract_color_token(styles.get("button", {}).get("border-color")) or "#d8dadd"

    fallbacks = {
        "text.body": (body_color, "inferred from computed body"),
        "text.heading": (heading_color, "inferred from computed h1/h2"),
        "text.muted": ("#777777", "default muted fallback"),
        "link.default": (link_color, "inferred from computed link"),
        "link.hover": (link_color, "inferred from computed link"),
        "button.primary.bg": (link_color, "inferred from link accent"),
        "button.primary.text": ("#ffffff", "default button text fallback"),
        "button.primary.border": (border_default, "inferred from computed button border"),
        "button.primary.hover.bg": (heading_color, "inferred from heading tone"),
        "surface.page": (page_bg, "inferred from computed body background"),
        "surface.section": ("#f5f5f7", "default section fallback"),
        "surface.card": ("#ffffff", "default card fallback"),
        "border.default": (border_default, "inferred from computed button border"),
        "icon.default": (heading_color, "inferred from heading tone"),
        "icon.accent": (link_color, "inferred from link accent"),
    }

    output: List[BrandRoleResult] = []
    for role, label in role_labels.items():
        if role in resolved:
            output.append(resolved[role])
            continue
        fallback_value, evidence = fallbacks[role]
        output.append(BrandRoleResult(role, label, fallback_value, evidence, "inferred"))

    return output


def render_font_row(label: str, styles: Dict[str, str]) -> str:
    family = clean_css_value(styles.get("font-family"))
    size = clean_css_value(styles.get("font-size")) or "1rem"
    weight = clean_css_value(styles.get("font-weight")) or "400"
    line_height = clean_css_value(styles.get("line-height")) or "1.4"
    name = primary_font_name(family)
    preview_family = family or "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif"
    return (
        '<div class="type-row">'
        f'<div class="type-name">{esc(label)} = {esc(name)}</div>'
        f'<div class="type-preview" style="font-family:{esc(preview_family)};font-size:{esc(size)};font-weight:{esc(weight)};line-height:{esc(line_height)}">{esc(label)} Sample</div>'
        f'<div class="role-evidence" style="font-family:{esc(preview_family)}">Aa Bb Cc 123</div>'
        "</div>"
    )


def render_color_item(label: str, value: str) -> str:
    return (
        '<div class="color-item">'
        f'<div class="swatch" style="background:{esc(value)}"></div>'
        f'<div class="color-label">{esc(label)}</div>'
        f'<div class="color-value">{esc(value)}</div>'
        "</div>"
    )


def render_brand_role_item(role: BrandRoleResult) -> str:
    badge = "Found" if role.status == "found" else "Inferred"
    return (
        '<div class="role-item">'
        f'<div class="role-swatch" style="background:{esc(role.value)}"></div>'
        f'<div class="role-title">{esc(role.label)} <span class="badge">{esc(badge)}</span></div>'
        f'<div class="role-value">{esc(role.value)}</div>'
        f'<div class="role-evidence">Evidence: {esc(role.evidence)}</div>'
        "</div>"
    )


def render_report(
    url: str,
    site_root_url: str,
    site_title: str,
    logo_url: Optional[str],
    logo_embed_src: Optional[str],
    styles: Dict[str, Dict[str, str]],
    font_face_css: str,
    brand_roles: List[BrandRoleResult],
) -> str:
    role_map = {role.key: role.value for role in brand_roles}
    body_hex = color_to_hex(role_map.get("text.body")) or "#000000"
    title_hex = color_to_hex(role_map.get("text.heading")) or body_hex
    link_hex = color_to_hex(role_map.get("link.default")) or body_hex
    typography_rows = [
        render_font_row("H1", styles.get("h1", {})),
        render_font_row("H2", styles.get("h2", {})),
        render_font_row("H3", styles.get("h3", {})),
        render_font_row("H4", styles.get("h4", {})),
        render_font_row("H5", styles.get("h5", {})),
        render_font_row("Body", styles.get("body", {})),
    ]

    quick_colors = [
        render_color_item("Body Color", body_hex),
        render_color_item("Title Color", title_hex),
        render_color_item("Link Color", link_hex),
    ]

    logo_block = '<div class="logo-missing">Logo not found</div>'
    logo_link = '<span class="logo-link muted">No direct logo URL found.</span>'
    if logo_url:
        logo_src = logo_embed_src or logo_url
        logo_block = f'<img class="logo-image" src="{esc(logo_src)}" alt="Site logo" />'
        logo_link = f'<a class="logo-link" href="{esc(logo_url)}" target="_blank" rel="noreferrer">{esc(logo_url)}</a>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CSS Grabber Report</title>
  <style>
    {font_face_css}
    :root {{
      --bg: #ffffff;
      --ink: #17181b;
      --muted: #6b6d73;
      --line: #d8dadd;
      --panel: #fafafb;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; }}
    .wrap {{ max-width: 1020px; margin: 40px auto 64px; padding: 0 22px; }}
    .toolbar {{ display: flex; justify-content: flex-end; margin-bottom: 10px; }}
    .print-btn {{ border: 1px solid var(--line); background: #fff; color: #1f2329; border-radius: 999px; padding: 8px 14px; font-size: 13px; cursor: pointer; }}
    .print-btn:hover {{ background: #f6f7f9; }}
    .line {{ border: 0; border-top: 1px solid var(--line); margin: 22px 0 28px; }}
    .section-title {{ margin: 0 0 16px; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size: 13px; font-weight: 600; }}
    .logo-wrap {{ text-align: center; padding: 8px 0 2px; }}
    .logo-image {{ max-width: 280px; max-height: 120px; width: auto; height: auto; object-fit: contain; }}
    .logo-missing {{ color: var(--muted); font-size: 14px; }}
    .logo-link {{ display: inline-block; margin-top: 10px; color: #0b63ce; font-size: 12px; word-break: break-all; text-decoration: none; }}
    .logo-link:hover {{ text-decoration: underline; }}
    .site-card {{ max-width: 520px; margin: 0 auto; text-align: center; border: 1px solid var(--line); padding: 14px 18px; }}
    .site-card-title {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 56px); font-weight: 600; letter-spacing: -0.02em; line-height: 1.05; }}
    .site-card-url {{ margin: 0; font-size: 13px; color: #2f3237; word-break: break-all; text-decoration: none; }}
    .site-card-url:hover {{ text-decoration: underline; }}
    .muted {{ color: var(--muted); }}

    .type-list {{ display: grid; gap: 12px; }}
    .type-row {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }}
    .type-name {{ font-size: 14px; margin-bottom: 7px; color: #2a2c30; }}

    .color-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .color-item {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }}
    .swatch {{ width: 100%; aspect-ratio: 4 / 3; border-radius: 8px; border: 1px solid rgba(0,0,0,.08); margin-bottom: 10px; }}
    .color-label {{ font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }}
    .color-value {{ font-size: 14px; }}

    .roles-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
    .role-item {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }}
    .role-swatch {{ width: 100%; height: 74px; border-radius: 8px; border: 1px solid rgba(0,0,0,.08); margin-bottom: 10px; }}
    .role-title {{ font-size: 13px; font-weight: 600; margin-bottom: 4px; }}
    .role-value {{ font-size: 13px; margin-bottom: 4px; }}
    .role-evidence {{ font-size: 12px; color: var(--muted); word-break: break-word; }}
    .badge {{ font-size: 11px; color: #7b7f87; font-weight: 500; }}
    @media print {{
      .toolbar {{ display: none; }}
      body {{ background: #fff; }}
      .wrap {{ margin: 0 auto; padding-top: 8px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="toolbar">
      <button class="print-btn" type="button" onclick="window.print()">Print | Save to PDF</button>
    </div>
    <div class="site-card">
      <h1 class="site-card-title">{esc(site_title)}</h1>
      <a class="site-card-url" href="{esc(site_root_url)}" target="_blank" rel="noreferrer">{esc(site_root_url)}</a>
      <div class="role-evidence">Page: <a class="site-card-url" href="{esc(url)}" target="_blank" rel="noreferrer">{esc(url)}</a></div>
    </div>
    <hr class="line" />

    <p class="section-title">Logo</p>
    <div class="logo-wrap">
      {logo_block}
      <br />
      {logo_link}
    </div>
    <hr class="line" />

    <section>
      <p class="section-title">Typography</p>
      <div class="type-list">{''.join(typography_rows)}</div>
    </section>

    <hr class="line" />
    <section>
      <p class="section-title">Colors</p>
      <div class="color-grid">{''.join(quick_colors)}</div>
    </section>

    <hr class="line" />
    <section>
      <p class="section-title">Brand Profile</p>
      <div class="roles-grid">{''.join(render_brand_role_item(role) for role in brand_roles)}</div>
    </section>
  </main>
</body>
</html>
"""


def classify_fetch_error(exc: Exception) -> Tuple[str, List[str]]:
    raw = str(exc)
    if isinstance(exc, HTTPError):
        if exc.code == 403:
            headers = exc.headers or {}
            if headers.get("cf-mitigated"):
                return (
                    "No dice: their bot wall stepped in (HTTP 403 challenge).",
                    [
                        "They are serving a challenge page instead of real content.",
                        "Try a different page on the same site.",
                        "Retry later or from another network.",
                    ],
                )
            return (
                "They said no (HTTP 403).",
                [
                    "This URL is blocking automated fetches.",
                    "Try another public page that is less protected.",
                    "Retry later in case the block is temporary.",
                ],
            )
        if exc.code == 429:
            return (
                "Too many requests, too fast (HTTP 429).",
                [
                    "Their server asked us to slow down.",
                    "Wait a few minutes, then run again.",
                    "You can also try a different URL.",
                ],
            )
        if exc.code >= 500:
            return (
                f"Their server is having a moment (HTTP {exc.code}).",
                [
                    "This is likely temporary on their side.",
                    "Retry once the site settles down.",
                ],
            )
        return (f"Request failed (HTTP {exc.code}).", ["Check the URL and try again."])

    if isinstance(exc, URLError):
        return (
            "Couldn’t reach that site from here.",
            [
                "DNS/network lookup failed for this domain.",
                "Double-check the URL spelling.",
                "Retry once network access is available.",
            ],
        )

    return (f"Something unexpected went wrong: {raw}", ["Retry and confirm the URL is publicly reachable."])


def render_error_report(url: str, summary: str, hints: List[str], raw_error: str) -> str:
    parsed = urlparse(url)
    site_root_url = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else url
    site_title = parsed.netloc or "Unknown Site"
    hint_items = "".join(f"<li>{esc(hint)}</li>" for hint in hints)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CSS Grabber Report - Error</title>
  <style>
    :root {{
      --bg: #ffffff;
      --ink: #17181b;
      --muted: #6b6d73;
      --line: #d8dadd;
      --panel: #fafafb;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; }}
    .wrap {{ max-width: 900px; margin: 40px auto 64px; padding: 0 22px; }}
    .toolbar {{ display: flex; justify-content: flex-end; margin-bottom: 10px; }}
    .print-btn {{ border: 1px solid var(--line); background: #fff; color: #1f2329; border-radius: 999px; padding: 8px 14px; font-size: 13px; cursor: pointer; }}
    .print-btn:hover {{ background: #f6f7f9; }}
    .site-card {{ max-width: 560px; margin: 0 auto; text-align: center; border: 1px solid var(--line); padding: 14px 18px; }}
    .site-card-title {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 44px); font-weight: 600; letter-spacing: -0.02em; line-height: 1.05; }}
    .site-card-url {{ margin: 0; font-size: 13px; color: #2f3237; word-break: break-all; text-decoration: none; }}
    .site-card-url:hover {{ text-decoration: underline; }}
    .line {{ border: 0; border-top: 1px solid var(--line); margin: 22px 0 28px; }}
    .section-title {{ margin: 0 0 16px; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size: 13px; font-weight: 600; }}
    .error-card {{ border: 1px solid #f2d4cf; background: #fff8f7; border-radius: 10px; padding: 14px; }}
    .error-title {{ color: var(--danger); font-weight: 700; margin-bottom: 6px; }}
    .raw {{ margin-top: 12px; padding: 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); color: #2a2c30; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; word-break: break-word; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin-bottom: 8px; }}
    @media print {{
      .toolbar {{ display: none; }}
      body {{ background: #fff; }}
      .wrap {{ margin: 0 auto; padding-top: 8px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="toolbar">
      <button class="print-btn" type="button" onclick="window.print()">Print | Save to PDF</button>
    </div>
    <div class="site-card">
      <h1 class="site-card-title">{esc(site_title)}</h1>
      <a class="site-card-url" href="{esc(site_root_url)}" target="_blank" rel="noreferrer">{esc(site_root_url)}</a>
      <div style="font-size:12px;color:#6b6d73;margin-top:6px;">Page: <a class="site-card-url" href="{esc(url)}" target="_blank" rel="noreferrer">{esc(url)}</a></div>
    </div>
    <hr class="line" />
    <p class="section-title">Report Status</p>
    <div class="error-card">
      <div class="error-title">Could not complete full extraction</div>
      <div>{esc(summary)}</div>
      <ul>{hint_items}</ul>
      <div class="raw">{esc(raw_error)}</div>
    </div>
  </main>
</body>
</html>
"""


def gather_css_sources(url: str, parser: SimpleHTMLIndex) -> List[Tuple[str, str]]:
    sources: List[Tuple[str, str]] = []
    for block in parser.style_blocks:
        sources.append((block, url))
    for href in parser.stylesheets:
        full = urljoin(url, href)
        try:
            sources.append((fetch_text(full), full))
        except Exception as exc:
            sources.append((f"/* failed to fetch {full}: {exc} */", full))
    return sources


def run(url: str, output: Path) -> None:
    html_text = fetch_text(url)
    parser = SimpleHTMLIndex()
    parser.feed(html_text)

    parsed = urlparse(url)
    site_root_url = f"{parsed.scheme}://{parsed.netloc}/"
    site_title = parser.site_name or parser.title_text or "Untitled Site"
    schema_logo_url = extract_schema_logo_url(html_text, url)
    logo_url = pick_logo_url(url, parser, schema_logo_url=schema_logo_url)
    logo_embed_src = image_data_uri(logo_url) if logo_url else None

    css_sources = gather_css_sources(url, parser)
    css_text = "\n".join(css for css, _ in css_sources)
    rules, _ = expand_css_rules(css_text)
    var_map = build_var_map(rules)
    styles = compute_styles(rules, parser.inline_styles, var_map, parser.first_element_attrs)

    font_entries: List[FontFaceEntry] = []
    for css, base in css_sources:
        font_entries.extend(extract_font_face_entries(css, base))

    used_families: Set[str] = set()
    for key in ("h1", "h2", "h3", "h4", "h5", "body"):
        fam = clean_css_value(styles.get(key, {}).get("font-family"))
        if not fam:
            continue
        first = fam.split(",", 1)[0].strip().strip("\"'")
        if first:
            used_families.add(first)

    inline_faces = inline_used_font_faces(font_entries, used_families)
    # Only include safely inlined faces. Raw third-party @font-face blocks can be malformed
    # (especially data URI/format edge cases) and break the report's style tag.
    font_face_css = inline_faces

    brand_roles = collect_brand_profile(rules, var_map, styles)

    report = render_report(url, site_root_url, site_title, logo_url, logo_embed_src, styles, font_face_css, brand_roles)
    output.write_text(report, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract logo, typography, and colors from a URL")
    ap.add_argument("url", help="Page URL (https://...) to inspect")
    ap.add_argument("-o", "--output", default="css-grabber-report.html", help="Output HTML file path")
    args = ap.parse_args()

    try:
        run(args.url, Path(args.output))
        print(f"Report written to {args.output}")
        return 0
    except Exception as exc:
        summary, hints = classify_fetch_error(exc)
        error_report = render_error_report(args.url, summary, hints, str(exc))
        Path(args.output).write_text(error_report, encoding="utf-8")
        print(f"Report written to {args.output} (error report)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
