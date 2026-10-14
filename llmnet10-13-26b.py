#!/usr/bin/env python3
"""
llmnet — a fake internet, one .llm domain at a time. (v3: profiles, local LLM support, comment policies)

  python3 llmnet.py serve                  # start the server
  python3 llmnet.py build apple.com.llm    # generate one site
  python3 llmnet.py build --seed           # starter set
  python3 llmnet.py build --all            # build referenced-but-unbuilt domains
  python3 llmnet.py map                    # print state
  python3 llmnet.py build apple.com.llm --rebuild

Examples:
  Anthropic:
    export ANTHROPIC_API_KEY=...
    python3 llmnet.py serve

  Ollama:
    export LLMNET_PROVIDER=ollama
    export LLMNET_MODEL=llama3.1
    export LLMNET_BASE_URL=http://localhost:11434
    python3 llmnet.py build apple.com.llm --rebuild

  OpenAI-compatible local server:
    export LLMNET_PROVIDER=openai
    export LLMNET_MODEL=local-model
    export LLMNET_BASE_URL=http://localhost:8080/v1
    python3 llmnet.py serve
"""
import argparse
import json
import os
import re
import shutil
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from html import escape as esc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.environ.get("LLMNET_ROOT", str(Path.home() / "fake-internet"))).expanduser()
SITES = ROOT / "sites"
PORT = int(os.environ.get("LLMNET_PORT", "8787"))
BIND = os.environ.get("LLMNET_BIND", "127.0.0.1")
ORIGIN = os.environ.get("LLMNET_ORIGIN", "").rstrip("/")

# ---------------------------------------------------------------- CONFIGURATION
# Edit these defaults or use LLMNET_* environment variables
PROVIDER = os.environ.get("LLMNET_PROVIDER", "openai").lower()
DEFAULT_BASE = {
    "anthropic": "https://api.anthropic.com",
    "openai": "http://localhost:8080/v1",
    "ollama": "http://localhost:11434",
}
if PROVIDER not in DEFAULT_BASE:
    PROVIDER = "anthropic"

BASE = (
    os.environ.get("LLMNET_BASE_URL")
    or os.environ.get("ANTHROPIC_BASE_URL")
    or DEFAULT_BASE[PROVIDER]
).rstrip("/")

MODEL = (
    os.environ.get("LLMNET_MODEL")
    or {
        "anthropic": "claude-haiku-4-5",
        "openai": "local-model",
        "ollama": "llama3.1",
    }[PROVIDER]
)

API_KEY = (
    os.environ.get("LLMNET_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)

TIMEOUT = int(os.environ.get("LLMNET_TIMEOUT", "600"))
RETRIES = int(os.environ.get("LLMNET_RETRIES", "3"))
TEMPERATURE = float(os.environ.get("LLMNET_TEMPERATURE", "1.0"))
MAX_TOKENS_DESIGN = int(os.environ.get("LLMNET_MAX_TOKENS_DESIGN", "4000"))
MAX_TOKENS_PAGE = int(os.environ.get("LLMNET_MAX_TOKENS_PAGE", "20000"))

DOMAIN_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.llm$"
)
BUILD_PATTERN = r"[a-z0-9](\.[a-z0-9-]+)*\.llm"

GEN_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
REG_LOCK = threading.Lock()

KINDS = ("news", "weather", "blog", "forum", "shop", "wiki", "product", "misc")
COMMENT_POLICIES = ("none", "info", "light", "reviews", "snarky", "troll", "forum")
ITEM_TYPES = ("article", "product", "press", "faq", "download", "video", "other")

# ---------------------------------------------------------------- utilities


def now():
    return datetime.now().isoformat(timespec="seconds")


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def fix_domain(d):
    d = str(d or "").lower().strip().lstrip("http:/").strip("/")
    d = d.split("/")[0].split(":")[0]
    if not d:
        return None
    if not d.endswith(".llm"):
        d += ".llm"
    return d if DOMAIN_RE.match(d) else None


# ---------------------------------------------------------------- state


def _load_state_nolock():
    p = ROOT / "state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"created": now(), "api_calls": 0, "tokens_out": 0, "log": []}


def _save_state_nolock(st):
    ROOT.mkdir(parents=True, exist_ok=True)
    st["log"] = st.get("log", [])[-40:]
    (ROOT / "state.json").write_text(json.dumps(st, indent=2))


def mutate_state(fn):
    with STATE_LOCK:
        st = _load_state_nolock()
        fn(st)
        _save_state_nolock(st)


def load_state():
    with STATE_LOCK:
        return _load_state_nolock()


def log(msg):
    def update(st):
        st.setdefault("log", []).append({"t": now(), "msg": msg})

    mutate_state(update)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bump_state(tokens):
    def update(st):
        st["api_calls"] = safe_int(st.get("api_calls", 0)) + 1
        st["tokens_out"] = safe_int(st.get("tokens_out", 0)) + safe_int(tokens)

    mutate_state(update)


# ---------------------------------------------------------------- registry


def _load_reg_nolock():
    p = ROOT / "registry.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_reg_nolock(reg):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "registry.json").write_text(json.dumps(reg, indent=2))


def load_reg():
    with REG_LOCK:
        return _load_reg_nolock()


def mutate_reg(fn):
    with REG_LOCK:
        reg = _load_reg_nolock()
        fn(reg)
        _save_reg_nolock(reg)


# ---------------------------------------------------------------- LLM provider


class APIError(RuntimeError):
    def __init__(self, msg, retryable=False):
        super().__init__(msg)
        self.retryable = retryable


RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504, 529}


def _endpoint(provider):
    base = BASE
    if provider == "anthropic":
        return base + "/v1/messages"
    if provider == "ollama":
        return base if base.endswith("/api/chat") else base.rstrip("/") + "/api/chat"
    if provider == "openai":
        return base if "/chat/completions" in base else base.rstrip("/") + "/chat/completions"
    return base


def _post_json(url, payload, headers):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        raise APIError(f"API error {e.code} from {BASE}: {detail or e.reason}", e.code in RETRYABLE_HTTP)
    except OSError as e:
        raise APIError(f"network error reaching {BASE}: {getattr(e, 'reason', None) or e}", True)


def llm_complete(system, user, max_tokens):
    """
    Provider-agnostic completion.
    Returns (text, output_tokens).
    """
    provider = PROVIDER
    last = None

    for attempt in range(1, RETRIES + 1):
        try:
            if provider == "anthropic":
                if not API_KEY:
                    raise APIError("ANTHROPIC_API_KEY is not set. Export it or use a local provider.")

                headers = {
                    "content-type": "application/json",
                    "x-api-key": API_KEY,
                    "anthropic-version": "2023-06-01",
                }
                payload = {
                    "model": MODEL,
                    "max_tokens": max_tokens,
                    "temperature": TEMPERATURE,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }
                raw = _post_json(_endpoint(provider), payload, headers)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise APIError(f"Anthropic returned non-JSON: {e}; first 200 chars: {raw[:200]}", True)

                text = "".join(
                    b.get("text", "")
                    for b in data.get("content", [])
                    if b.get("type") == "text"
                )
                return text.strip(), safe_int(data.get("usage", {}).get("output_tokens", 0))

            elif provider == "openai":
                headers = {"content-type": "application/json"}
                if API_KEY:
                    headers["authorization"] = f"Bearer {API_KEY}"

                payload = {
                    "model": MODEL,
                    "max_tokens": max_tokens,
                    "temperature": TEMPERATURE,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                raw = _post_json(_endpoint(provider), payload, headers)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise APIError(f"OpenAI-compatible returned non-JSON: {e}; first 200 chars: {raw[:200]}", True)

                if not data.get("choices"):
                    raise APIError("OpenAI-compatible response missing choices.", True)

                text = data["choices"][0].get("message", {}).get("content") or ""
                return text.strip(), safe_int(data.get("usage", {}).get("completion_tokens", 0))

            elif provider == "ollama":
                headers = {"content-type": "application/json"}
                payload = {
                    "model": MODEL,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": TEMPERATURE,
                    },
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                raw = _post_json(_endpoint(provider), payload, headers)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise APIError(f"Ollama returned non-JSON: {e}; first 200 chars: {raw[:200]}", True)

                text = data.get("message", {}).get("content") or ""
                return text.strip(), safe_int(
                    data.get("eval_count", 0) or data.get("usage", {}).get("completion_tokens", 0)
                )

            else:
                raise APIError(f"Unknown LLMNET_PROVIDER: {provider}. Use anthropic, openai, or ollama.", False)

        except APIError as e:
            last = e
            if not e.retryable or attempt >= RETRIES:
                raise RuntimeError(str(e))
            wait = 2 * attempt
            log(f"  LLM hiccup (attempt {attempt}/{RETRIES}): {e}")
            log(f"  retrying in {wait}s ...")
            time.sleep(wait)
            continue

    raise RuntimeError(f"gave up after {RETRIES} attempts: {last}")


def extract_json(raw):
    """Robust JSON extractor for LLM metadata responses."""
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        raise RuntimeError(f"No JSON object found in model output: {raw[:200]}")

    json_str = m.group(0)
    try:
        return json.loads(json_str, strict=False)
    except Exception:
        # Fallback fix for bad escape backslashes and trailing commas
        cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON metadata: {e}\nRaw start: {raw[:200]}")


def extract_html(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)

    m = re.search(r"(?i)<!doctype html>|<html", t)
    start = m.start() if m else 0
    low = t.lower()
    end = low.rfind("</html>")
    if end != -1:
        end += len("</html>")
    else:
        end = len(t)

    out = t[start:end].strip()
    if out and "</html>" not in out.lower():
        out += "\n</body>\n</html>"

    if "<html" not in out.lower() and "<!doctype" not in out.lower():
        raise RuntimeError("No HTML document found in model output. Retry.")
    return out


# ---------------------------------------------------------------- site profiles


SITE_PROFILES = {
    "apple": {
        "kind": "product",
        "comment_policy": "none",
        "community_tone": "clean, premium, information-first; no jokes, no snark, no trolling, no user comments",
        "must_include": [
            "macbook-pro.html",
            "iphone.html",
            "ipad.html",
            "airpods.html",
            "support.html",
        ],
        "forbidden": [
            "jokes",
            "snark",
            "trolling",
            "comment links",
            "real logos",
            "exact verbatim marketing copy from the real company",
        ],
        "notes": "Make an Apple-inspired product site: minimal white background, large product names, simple nav, spec tables, Buy/Compare/Support style CTAs.",
    },
    "fark": {
        "kind": "news",
        "comment_policy": "snarky",
        "community_tone": "snarky, fast, link-index, joke culture, playful trolling, lawyer in the comments",
        "must_include": [
            "dense front page",
            "section rails",
            "comment counts",
        ],
        "forbidden": [
            "clean modern magazine layout",
            "no comment links on stories",
        ],
        "notes": "Dense, busy, funny link index.",
    },
    "weather": {
        "kind": "weather",
        "comment_policy": "light",
        "community_tone": "farmers almanac, Denver historical outlook, a little folk humor, no heavy trolling",
        "must_include": [
            "denver-historical-outlook.html",
        ],
        "forbidden": [
            "heavy snark",
            "trolling",
        ],
        "notes": "Denver, Colorado historical/climatological star page.",
    },
}


def profile_for(domain):
    for key, prof in SITE_PROFILES.items():
        if key in domain:
            return prof
    return {}


SITE_HINTS = {
    "apple.com.llm": (
        "Apple-inspired product site: clean, minimal, product-first. "
        "Information-only. No snark. No trolling. No comment links."
    ),
    "fark.com.llm": (
        "Snarky news aggregator: dense link index, funny one-line stories, "
        "source links, comment counts, playful but safe comment culture."
    ),
    "weather.com.llm": (
        "Fictional national weather service. Star city: Denver, Colorado. "
        "Use historical/climatological style: monthly averages, seasonal records, "
        "7-day outlook framed as what Denver's history does."
    ),
}


def hint_for(domain):
    d = domain
    if any(w in d for w in ("apple", "product", "catalog")):
        return "the name suggests a product catalog site — clean, information-first, minimal, no snark"
    if any(w in d for w in ("weather", "forecast", "climate")):
        return "the name suggests a weather service — star city current conditions + historical outlook"
    if any(w in d for w in ("fark", "news", "press", "report", "post", "daily")):
        return "the name suggests a snarky news aggregator or link site"
    if any(w in d for w in ("wiki", "pedia", "index")):
        return "the name suggests a wiki / encyclopedia"
    if any(w in d for w in ("shop", "store", "mart", "market", "supply")):
        return "the name suggests a shop"
    if any(w in d for w in ("forum", "board", "chat", "bbs")):
        return "the name suggests a forum"
    if any(w in d for w in ("blog", "diary", "journal", "notes", "musing")):
        return "the name suggests a personal blog"
    return "no strong hint — choose the kind that best fits the name"


# ---------------------------------------------------------------- prompts


FONT_LINKS = {
    "A": '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Spectral:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">',
    "B": '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;1,8..60,400&display=swap" rel="stylesheet">',
    "C": '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,900&family=Inter:wght@400;600&display=swap" rel="stylesheet">',
    "D": '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">',
}


DESIGN_SYSTEM = """You are the DESIGNER for ".llm", a private, entirely fictional internet.
Given a domain name, design its brand identity, color palette, comment rules, items/articles list, and outbound links.

OUTPUT FORMAT (STRICT):
Output a single valid JSON object representing the site design metadata. DO NOT include raw HTML in this JSON object.

JSON SCHEMA:
{
  "site_name": "Brand Name",
  "tagline": "Short in-voice tagline",
  "kind": "news|weather|blog|forum|shop|wiki|product|misc",
  "palette": {"bg": "#f4f3ee", "paper": "#ffffff", "ink": "#171b18", "accent": "#c9452c", "soft": "#7d857e"},
  "fonts": "A|B|C|D",
  "layout_notes": "One sentence describing visual identity",
  "comment_policy": "none|info|light|reviews|snarky|troll|forum",
  "community_tone": "Description of comment culture or 'none'",
  "commenters": ["handle1", "handle2"],
  "items": [
    {
      "slug": "kebab-case-slug",
      "title": "Item Title",
      "blurb": "One sentence blurb",
      "meta": "Flavor line",
      "type": "article|product|press|faq|download|video|other",
      "has_comments": true,
      "comments": 5,
      "source": "other-invented-domain.llm"
    }
  ],
  "links": [{"domain": "invented-domain.llm", "label": "Short Label"}]
}
"""

INDEX_HTML_SYSTEM = """You are the FRONT PAGE BUILDER for an existing site in the ".llm" fictional internet.
Given the site's DESIGN metadata, build a complete, hand-crafted front page.

LAYOUT GUIDELINES BY KIND:
- product: Minimal Apple-inspired layout: clean sticky nav, large titles, spec tables, feature grid, Buy/Compare CTAs.
- news/aggregator: Dense link index with numbered rows, section rails, and small meta lines.
- weather: Big weather panel, 7-day strip, records table, long-range outlook block.
- blog: Big author masthead, continuous story list, oversized dates.
- forum: High-density thread table with user badges and reply counts.

TECHNICAL REQUIREMENTS:
- Output ONE valid complete HTML5 document (doctype, head, body, inline CSS in <style>).
- Paste the exact font link tag specified in the design verbatim into <head>.
- Every link uses ~SRV~ origin:
    same site: href="~SRV~/domain.llm/index.html" or href="~SRV~/domain.llm/slug.html"
    other site: href="~SRV~/otherdomain.llm/"
- Include header, footer with "fiction" mark, and nearby links. Add prefers-reduced-motion CSS media query.
- Output HTML ONLY. No JSON wrapper, no markdown fences, no commentary.
"""

PAGE_SYSTEM = """You are the PAGE GENERATOR for one page of an existing site in the ".llm" fictional internet.
You are given the site's DESIGN (identity, palette, fonts, voice) and the ITEM this page is about.

Write ONE complete standalone HTML5 page: doctype, head, body, ALL CSS inline in a <style> tag, no external JavaScript, no network images (CSS and unicode visuals only). It must be visually consistent with the site's design but tailored to this page type.

LINK RULES — every link uses the ~SRV~ origin:
  same site:  href="~SRV~/fark.com.llm/index.html"
              href="~SRV~/fark.com.llm/city-cat-elected-mayor.html"
              href="~SRV~/fark.com.llm/city-cat-elected-mayor/comments.html"
  other site: href="~SRV~/mayorwatch.news.llm/"

Include the site's nav (Home / this page / About / Links) in the site's style, and the site's footer (voice note + "fiction" mark).
Add a prefers-reduced-motion media query.
Output the HTML document ONLY. No JSON, no markdown fences, no commentary.
"""


def design_user_prompt(domain, brief, profile):
    prof = json.dumps(profile, ensure_ascii=False, indent=2) if profile else "no special profile"
    return f"""HOST DOMAIN: {domain}
SITE BRIEF: {brief}
SITE PROFILE (strict): {prof}
Create the design schema JSON metadata now.
"""


def _head_line(design, domain):
    return (
        f"SITE DOMAIN: {domain}\n"
        f"PASTE VERBATIM IN HEAD: {FONT_LINKS.get(design.get('fonts'), FONT_LINKS['A'])}\n"
        f"SITE: {design.get('site_name','')} — {design.get('tagline','')} — {design.get('layout_notes','')}\n"
        f"PALETTE: {json.dumps(design.get('palette', {}))}\n"
        f"COMMENT POLICY: {design.get('comment_policy', 'light')}\n"
        f"COMMUNITY TONE: {design.get('community_tone', 'fictional and safe')}\n"
    )


def content_prompt(design, domain, item, items):
    slug = item.get("slug") or "page"
    typ = item.get("type") or "article"
    policy = design.get("comment_policy", "light")
    has_cmt = bool(item.get("has_comments"))
    if policy == "none":
        has_cmt = False

    others = [
        f"{i.get('title')} at ~SRV~/{domain}/{i.get('slug')}.html"
        for i in items
        if i.get("slug") != slug
    ][:3]
    others = "; ".join(others) or "none"

    src = item.get("source")
    if src:
        src_line = (
            f'Include a small "source" panel linking to ~SRV~/{src}/ — say this .llm page links to it as the source.'
        )
    else:
        src_line = "This page has no external source; say it was produced in-house by the site."

    if typ == "product":
        return f"""PAGE TYPE: product page.
{_head_line(design, domain)}
PRODUCT SLUG: {slug}
PRODUCT TITLE: {item.get('title')}
PRODUCT BLURB: {item.get('blurb')}
PRODUCT META: {item.get('meta')}

Write a clean product page: hero, concise value proposition, key specs table, feature cards, compatibility/accessories, and CTA buttons (Buy, Compare, Support).
Do NOT include jokes, snark, trolling, or user comments.
{src_line}
Do NOT include a comment link or comment count. This page is information-only.
Related products: {others}
"""

    if typ == "press":
        return f"""PAGE TYPE: press page.
{_head_line(design, domain)}
PRESS SLUG: {slug}
PRESS TITLE: {item.get('title')}
PRESS BLURB: {item.get('blurb')}
PRESS META: {item.get('meta')}

Write a fictional press release in the site's voice. Keep it specific and plausible, but clearly fictional.
{src_line}
Comment link instruction: {("Include a prominent 'read the comments' link to ~SRV~/" + domain + "/" + slug + "/comments.html labeled with the comment count.") if has_cmt else "Do NOT include a comment link or comment count."}
Related pages: {others}
"""

    if typ == "faq":
        return f"""PAGE TYPE: FAQ page.
{_head_line(design, domain)}
FAQ SLUG: {slug}
FAQ TITLE: {item.get('title')}
FAQ BLURB: {item.get('blurb')}
FAQ META: {item.get('meta')}

Write a useful FAQ page with 4-8 questions and answers in the site's voice.
{src_line}
Comment link instruction: {("Include a prominent 'read the comments' link to ~SRV~/" + domain + "/" + slug + "/comments.html labeled with the comment count.") if has_cmt else "Do NOT include a comment link or comment count."}
Related pages: {others}
"""

    comment_line = (
        f'Include a prominent "read the comments" link to ~SRV~/{domain}/{slug}/comments.html labeled with the comment count ({safe_int(item.get("comments", 0))}).'
        if has_cmt
        else "Do NOT include a comment link or comment count. This page is information-only."
    )

    return f"""PAGE TYPE: article/content page.
{_head_line(design, domain)}
ARTICLE SLUG: {slug}
ARTICLE TITLE: {item.get('title')}
ARTICLE BLURB: {item.get('blurb')}
ARTICLE META: {item.get('meta')}
COMMENT COUNT: {safe_int(item.get('comments', 0))}

Write 2-6 paragraphs of article body in the site's voice (specific, funny if the tone calls for it, fictional).
Include a byline/meta line.
{src_line}
{comment_line}
Add a related-stories block with: {others}
"""


def comments_prompt(design, domain, item):
    policy = design.get("comment_policy", "light")
    if policy == "none":
        policy = "light"

    tone = design.get("community_tone") or "fictional community"
    n = max(7, min(12, safe_int(item.get("comments"), 7)))

    names = [str(x).strip() for x in (design.get("commenters") or []) if str(x).strip()]
    if not names:
        names = ["anon_67", "mod_" + slugify(domain)[:12], "lawyersonsite"]

    if policy in ("snarky", "troll"):
        style = (
            "Use the site's snark/troll culture: quick one-liners, bait, playful irony, one lawyer, one not-OP, mild trolling. "
            "Keep it fictional and safe: no slurs, no harassment, no real private individuals, no real doxxing."
        )
    elif policy == "reviews":
        style = (
            "Use review culture: star ratings, pros/cons, product questions, a moderator note. "
            "Keep it helpful, fictional, and safe."
        )
    elif policy == "forum":
        style = (
            "Use forum culture: thread replies, user badges, helpful/silly replies, one moderator. "
            "Keep it fictional and safe."
        )
    else:
        style = (
            "Use a light comment culture: a few informed comments, one joke, one moderator note. "
            "Keep it fictional and safe."
        )

    return f"""PAGE TYPE: the comment section for ONE specific article.
{_head_line(design, domain)}
ARTICLE SLUG: {item.get('slug')}
ARTICLE TITLE: {item.get('title')}
ARTICLE BLURB: {item.get('blurb')}
COMMUNITY TONE: {tone}
STYLE: {style}

Write {n} UNIQUE comments, each one specifically about THIS article's subject (no generic filler).
Vary the karma of each.
Include exactly one staff comment signed mod@{domain}.
Use these commenter handles where they fit: {json.dumps(names)}

STRUCTURE (strict):
1. Put the comments inside a single <div id="thread">.
2. Then a line like: Showing {n} of {safe_int(item.get('comments'), n)} — the rest are in a drawer somewhere.
3. Then a comment form: <form onsubmit="return post()"> containing a <textarea id="cbox" rows="3"> and a Post button.
4. Finally include an inline <script> defining window.post: on submit, if #cbox is non-empty, append a new comment element (styled like the others) to #thread named 'you (guest)' with a NEW badge, clear the box, and after about 900 ms append one short canned auto-reply chosen from a small in-page array of replies in the site's voice (with a REPLY badge). Use only DOM APIs; set user text via textContent, never innerHTML.
5. Include a back link to the article: ~SRV~/{domain}/{item.get('slug')}.html
"""


def about_prompt(design, domain, items):
    n_items = len(items)
    n_cmt = sum(safe_int(i.get("comments", 0)) for i in items)
    policy = design.get("comment_policy", "light")
    policy_line = (
        "Comments are intentionally minimal or absent on many pages."
        if policy in ("none", "info")
        else "Some pages have comment sections; they are fictional and part of the site's culture."
    )
    return f"""PAGE TYPE: about page.
{_head_line(design, domain)}
Write a short in-voice about page: the site's voice note, what it is (fiction, generated by the .llm internet, saved to disk), a stats row ({n_items} stories, {n_cmt} comments, 100% fiction), and a disclaimer.
{policy_line}
Include a back link to ~SRV~/{domain}/index.html.
"""


def links_prompt(design, domain):
    rows = "; ".join(f"{l.get('label')} at ~SRV~/{l.get('domain')}/" for l in design.get("links", [])[:16])
    return f"""PAGE TYPE: the links page.
{_head_line(design, domain)}
List every one of these as its own row or card with a short label: {rows}
Frame them as doors: unvisited ones build themselves when you arrive.
Include a back link to ~SRV~/{domain}/index.html.
"""


# ---------------------------------------------------------------- generation


def generate_design(domain, brief):
    """Step 1: Generate metadata JSON. Step 2: Generate Front Page HTML directly."""
    profile = profile_for(domain)
    
    # Step 1: Design JSON metadata
    raw_design, tokens_design = llm_complete(
        DESIGN_SYSTEM,
        design_user_prompt(domain, brief, profile),
        MAX_TOKENS_DESIGN,
    )
    data = extract_json(raw_design)
    design = normalize_design(data, domain)

    # Step 2: Front page HTML generation
    index_prompt = (
        f"{_head_line(design, domain)}\n"
        f"KIND: {design.get('kind')}\n"
        f"ITEMS: {json.dumps(design.get('items', []))}\n"
        f"LINKS: {json.dumps(design.get('links', []))}\n\n"
        f"Build the full, complete front page HTML5 document for {domain} following this design schema."
    )
    raw_html, tokens_html = llm_complete(INDEX_HTML_SYSTEM, index_prompt, MAX_TOKENS_PAGE)
    html = extract_html(raw_html)

    return design, html, tokens_design + tokens_html


def normalize_design(design, domain):
    profile = profile_for(domain)

    design["site_name"] = str(design.get("site_name") or domain.split(".")[0]).strip()[:40]
    design["tagline"] = str(design.get("tagline") or "").strip()[:140]
    design["layout_notes"] = str(design.get("layout_notes") or "").strip()[:220]
    design["voice_note"] = str(design.get("voice_note") or design.get("tagline") or "").strip()[:220]

    kind = design.get("kind")
    if kind not in KINDS:
        kind = profile.get("kind") if profile.get("kind") in KINDS else "misc"
    design["kind"] = kind

    design["fonts"] = design.get("fonts") if design.get("fonts") in FONT_LINKS else "A"

    defaults = {"bg": "#f4f3ee", "paper": "#ffffff", "ink": "#171b18", "accent": "#c9452c", "soft": "#7d857e"}
    p = design.get("palette") if isinstance(design.get("palette"), dict) else {}
    hexre = re.compile(r"^#[0-9a-fA-F]{6}$")
    design["palette"] = {
        k: (str(p[k]) if k in p and hexre.match(str(p[k])) else defaults[k]) for k in defaults
    }

    policy = design.get("comment_policy")
    if policy not in COMMENT_POLICIES:
        policy = profile.get("comment_policy")
    if policy not in COMMENT_POLICIES:
        policy = "none" if kind == "product" else "light"
    design["comment_policy"] = policy

    tone_default = (
        "clean, information-first, no user comments"
        if kind == "product"
        else "fictional, opinionated, safe"
    )
    design["community_tone"] = str(
        design.get("community_tone") or profile.get("community_tone") or tone_default
    ).strip()[:400]

    if policy == "none":
        design["commenters"] = []
    else:
        cms = design.get("commenters")
        design["commenters"] = (
            [str(c).strip()[:30] for c in cms if isinstance(c, str) and c.strip()][:8]
            if isinstance(cms, list)
            else []
        )

    items, seen = [], set()
    for it in (design.get("items") if isinstance(design.get("items"), list) else [])[:15]:
        if not isinstance(it, dict):
            continue

        title = str(it.get("title") or "Untitled")[:160]
        slug = slugify(it.get("slug") or title)[:48] or "item"
        base, n = slug, 1
        while slug in seen:
            slug, n = f"{base}-{n}", n + 1
        seen.add(slug)

        typ = it.get("type")
        if typ not in ITEM_TYPES:
            typ = "product" if kind == "product" else "article"

        hc = it.get("has_comments")
        if hc is None:
            hc = policy != "none"
        hc = bool(hc)
        if policy == "none":
            hc = False

        comments = safe_int(it.get("comments", 0))
        comments = max(0, min(999, comments))
        if not hc:
            comments = 0

        src = fix_domain(it.get("source") or "")
        items.append(
            {
                "slug": slug,
                "title": title,
                "blurb": str(it.get("blurb") or "")[:180],
                "meta": str(it.get("meta") or "")[:120],
                "type": typ,
                "has_comments": hc,
                "comments": comments,
                "source": src if (src and src != domain) else None,
            }
        )

    if not items:
        if kind == "product":
            items = [
                {
                    "slug": "featured-product",
                    "title": "Featured product",
                    "blurb": "A clean product page.",
                    "meta": "product 001",
                    "type": "product",
                    "has_comments": False,
                    "comments": 0,
                    "source": None,
                }
            ]
        else:
            items = [
                {
                    "slug": "the-first-entry",
                    "title": "The first entry",
                    "blurb": "Everything starts somewhere.",
                    "meta": "entry 001",
                    "type": "article",
                    "has_comments": policy != "none",
                    "comments": 3 if policy != "none" else 0,
                    "source": None,
                }
            ]

    links, lseen = [], set()
    for d in (design.get("links") if isinstance(design.get("links"), list) else [])[:20]:
        if not isinstance(d, dict):
            continue
        dd = fix_domain(d.get("domain", ""))
        if not dd or dd == domain or dd in lseen:
            continue
        lseen.add(dd)
        links.append({"domain": dd, "label": str(d.get("label") or dd)[:60]})
    design["links"] = links

    return design


FALLBACK_COMMENT_JS = r"""
const NAMES=__NAMES__;
const REPLIES=[
 "Cromulent. This is exactly what the fake internet is for.",
 "I am a lawyer and I can confirm: this thread is fiction.",
 "Not OP, but this is the best post on the site.",
 "I read the whole thing. My takeaway is: it is all of it.",
 "This comment was written by a bot that believes in you.",
 "I for one welcome our robot overlords and thier fake internet."
];
function avc(s){const c=['#c9452c','#1f6f54','#2b6cb0','#7a5aa8','#b07818'];return c[s.charCodeAt(0)%c.length];}
function add(name,text,badge){
  const box=document.getElementById('thread');
  const el=document.createElement('div');el.className='cm-fallback';
  el.style.cssText='padding:14px 0;border-bottom:1px dashed #999';
  const top=document.createElement('div');top.style.cssText='display:flex;gap:10px;align-items:center;margin-bottom:6px;flex-wrap:wrap';
  const av=document.createElement('span');av.style.cssText='width:28px;height:28px;border-radius:3px;display:inline-grid;place-items:center;color:#fff;font:600 11px monospace';
  av.style.background=avc(name);av.textContent=name.replace(/[^a-z]/gi,'').slice(0,2).toUpperCase()||'??';
  const b=document.createElement('b');b.textContent=name;
  top.append(av,b);
  if(badge){const s=document.createElement('span');s.textContent=badge;s.style.cssText='font:700 9px monospace;letter-spacing:.12em;border:1px solid #999;padding:2px 6px';top.append(s);}
  const kr=document.createElement('span');kr.textContent='karma 0';kr.style.cssText='font:11px monospace;color:#888';
  top.append(kr);
  const tx=document.createElement('div');tx.textContent=text;
  el.append(top,tx);box.appendChild(el);
  el.scrollIntoView({behavior:'smooth',block:'center'});
}
window.post=function(){
  const t=document.getElementById('cbox').value.trim();
  if(!t){return false;}
  add('you (guest)',t,'NEW');
  document.getElementById('cbox').value='';
  setTimeout(function(){add(NAMES[(Math.random()*NAMES.length)|0],REPLIES[(Math.random()*REPLIES.length)|0],'REPLY');},900);
  return false;
};
"""


def ensure_comment_js(html, design):
    if 'id="cbox"' not in html and "id='cbox'" not in html:
        return html
    if "window.post" in html or "function post(" in html:
        return html

    names = json.dumps(design.get("commenters", [])[:8] or ["anon_67", "lawyersonsite"])
    js = FALLBACK_COMMENT_JS.replace("__NAMES__", names)
    if "</body>" in html:
        return html.replace("</body>", f"<script>{js}</script></body>", 1)
    return html + f"<script>{js}</script></body></html>"


def inject_favicon(html, design):
    if 'rel="icon"' in html or "rel='icon'" in html:
        return html
    accent = (design.get("palette") or {}).get("accent", "#c9452c")
    a = urllib.parse.quote(accent)
    fav = (
        '<link rel="icon" href="data:image/svg+xml,'
        "%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E"
        f"%3Crect width=%2732%27 height=%2732%27 fill=%27{a}%27/%3E"
        "%3Ctext x=%2716%27 y=%2723%27 font-size=%2718%27 text-anchor=%27middle%27 fill=%27%23fff%27 font-family=%27monospace%27%3E~%3C/text%3E%3C/svg%3E\">"
    )
    i = html.lower().find("<head")
    if i != -1:
        j = html.find(">", i)
        if j != -1:
            return html[: j + 1] + fav + html[j + 1 :]
    return html


# ---------------------------------------------------------------- building


def referenced_domains(design, self_domain):
    out = {}
    for it in design.get("items", []):
        if it.get("source") and it["source"] != self_domain:
            out.setdefault(it["source"], f'source for "{it["title"]}"')
    for l in design.get("links", []):
        if l.get("domain") and l["domain"] != self_domain:
            out.setdefault(l["domain"], f'listed in the links page as "{l.get("label", "")}"')
    return list(out.items())


def ensure_domain(domain):
    domain = fix_domain(domain)
    if not domain:
        raise RuntimeError(f"'{domain}' is not a valid .llm domain")

    ddir = SITES / domain
    if (ddir / "design.json").exists() and (ddir / "index.html").exists():
        return ddir, False

    with GEN_LOCK:
        if (ddir / "design.json").exists() and (ddir / "index.html").exists():
            return ddir, False

        reg = load_reg()
        ctx = reg.get(domain, {})
        nref = len(ctx.get("referenced_by", []))
        log(f"building {domain} …" + (f" (already referenced by {nref} site(s))" if nref else ""))

        brief = SITE_HINTS.get(domain) or hint_for(domain)
        profile = profile_for(domain)
        if profile:
            brief += "\nSITE PROFILE (strict): " + json.dumps(profile, ensure_ascii=False, indent=2)

        if ctx.get("referenced_by"):
            brief += (
                " ALSO: this domain is already referenced in the .llm internet as: "
                + "; ".join(
                    f"by {r.get('domain')} ({r.get('note', '')})" for r in ctx["referenced_by"][:4]
                )
                + " — include that content (in your voice) among your items."
            )

        design, index_html, tokens = generate_design(domain, brief)
        design = normalize_design(design, domain)
        index_html = inject_favicon(index_html, design)

        ddir.mkdir(parents=True, exist_ok=True)
        (ddir / "design.json").write_text(json.dumps(design, indent=2, ensure_ascii=False))
        (ddir / "index.html").write_text(index_html, encoding="utf-8")

        def update(reg):
            e = reg.setdefault(domain, {"first_seen": now()})
            e.update({"status": "built", "built_at": now(), "site_name": design.get("site_name")})
            for d, note in referenced_domains(design, domain):
                en = reg.setdefault(d, {"first_seen": now(), "status": "referenced"})
                if en.get("status") != "built":
                    en.setdefault("referenced_by", []).append({"domain": domain, "note": note})

        mutate_reg(update)
        bump_state(tokens)
        n_items = len(design.get("items", []))
        log(f"  → {domain}: front page + design saved ({n_items} items)")
        return ddir, True


def classify(path, design):
    items = {i.get("slug"): i for i in design.get("items", []) if i.get("slug")}

    if path.endswith("/comments.html"):
        slug = path[:-len("/comments.html")]
        if slug in items:
            if items[slug].get("has_comments", True):
                return ("comments", items[slug])
            return ("comments_disabled", items[slug])
        # Allow dynamic comments generation for unlisted slugs
        dynamic_item = {
            "slug": slug,
            "title": slug.replace("-", " ").title(),
            "blurb": f"Discussion regarding {slug.replace('-', ' ')}.",
            "meta": "discussion",
            "type": "article",
            "has_comments": True,
            "comments": 5,
        }
        return ("comments", dynamic_item)

    if path.endswith(".html"):
        slug = path[:-5]
        if slug in ("about", "links"):
            return (slug, None)
        if slug in items:
            return ("article", items[slug])
        
        # FIX: On-the-fly item creation for links generated in HTML body
        dynamic_item = {
            "slug": slug,
            "title": slug.replace("-", " ").title(),
            "blurb": f"An article about {slug.replace('-', ' ')}.",
            "meta": "breaking news",
            "type": "article",
            "has_comments": design.get("comment_policy") != "none",
            "comments": 3 if design.get("comment_policy") != "none" else 0,
            "source": None,
        }
        return ("article", dynamic_item)

    return None


def generate_page(design, domain, kind, item):
    if kind == "comments":
        up = comments_prompt(design, domain, item)
    elif kind == "about":
        up = about_prompt(design, domain, design.get("items", []))
    elif kind == "links":
        up = links_prompt(design, domain)
    else:
        up = content_prompt(design, domain, item, design.get("items", []))

    raw, tokens = llm_complete(PAGE_SYSTEM, up, MAX_TOKENS_PAGE)
    return extract_html(raw), tokens


def ensure_page(domain, rel, kind, item):
    ddir = SITES / domain
    target = ddir / rel
    if target.exists():
        return target

    with GEN_LOCK:
        if target.exists():
            return target

        design = json.loads((ddir / "design.json").read_text())
        log(f"generating {domain}/{rel} …")
        
        # Ensure the item is recorded in design.json if created dynamically
        slug = item.get("slug")
        existing_slugs = {i.get("slug") for i in design.get("items", [])}
        if slug and slug not in existing_slugs and kind == "article":
            design.setdefault("items", []).append(item)
            (ddir / "design.json").write_text(json.dumps(design, indent=2, ensure_ascii=False))

        html, tokens = generate_page(design, domain, kind, item)
        if kind == "comments":
            html = ensure_comment_js(html, design)
        html = inject_favicon(html, design)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        bump_state(tokens)
        log(f"  → {domain}/{rel} saved")
        return target


# ---------------------------------------------------------------- hub / 404 / disabled comments


HUB_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b2e21;color:#eaf3ec;font-family:Georgia,'Times New Roman',serif;font-size:16px;line-height:1.6}
.wrap{max-width:1000px;margin:0 auto;padding:0 22px}
.top{background:#062018;border-bottom:1px solid #1d4636;padding:10px 0;font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.14em;color:#7fb79a;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
h1{font-size:2.2rem;letter-spacing:-.01em;margin-bottom:6px}
.sub{color:#9fc4ae;font-style:italic;margin-bottom:28px;max-width:70ch}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:30px}
.stat{border:1px solid #1d4636;border-radius:3px;padding:14px 18px;min-width:130px}
.stat b{display:block;font-size:1.7rem;font-family:ui-monospace,Menlo,monospace;color:#35d07a}
.stat span{font-family:ui-monospace,Menlo,monospace;font-size:10px;letter-spacing:.14em;color:#7fb79a}
form{display:flex;gap:10px;margin-bottom:30px;flex-wrap:wrap}
input{flex:1;min-width:220px;background:#0f3527;border:1px solid #245a44;color:#dff0e4;font-family:ui-monospace,Menlo,monospace;padding:12px 14px;border-radius:2px;font-size:14px}
input::placeholder{color:#5f9278}
button{background:#35d07a;color:#06231a;border:none;font:600 12px ui-monospace,Menlo,monospace;letter-spacing:.1em;text-transform:uppercase;padding:12px 20px;border-radius:2px;cursor:pointer}
button:hover{filter:brightness(1.1)}
table{width:100%;border-collapse:collapse;margin-bottom:34px}
td{padding:10px 8px;border-bottom:1px solid #174534;font-size:14px;vertical-align:top;text-align:left}
td.d{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:#eaf3ec}
tr.u td.d{color:#d9a856}
a{color:#35d07a}
h2{font-size:1rem;font-family:ui-monospace,Menlo,monospace;letter-spacing:.16em;text-transform:uppercase;color:#7fb79a;margin:26px 0 10px}
pre{background:#062018;border:1px solid #1d4636;padding:14px;font-size:12px;overflow:auto;border-radius:2px}
ul{margin-left:18px}li{margin:5px 0;color:#9fc4ae;font-size:14px}
"""


def hub_html():
    reg, st = load_reg(), load_state()
    built = sorted(d for d, v in reg.items() if v.get("status") == "built")
    unbuilt = sorted(d for d, v in reg.items() if v.get("status") != "built")
    n_pages = 0
    for d in reg:
        dd = SITES / d
        if dd.exists():
            n_pages += len(list(dd.rglob("*.html")))

    def row(d, v, ok):
        refs = v.get("referenced_by") or []
        note = " · seen from " + ", ".join(r.get("domain", "?") for r in refs[:2]) if refs else ""
        name = f" “{v.get('site_name')}”" if v.get("site_name") else ""
        return (
            f"<tr class='{'u' if not ok else ''}'><td class='d'>{esc(d)}</td>"
            f"<td>{'built' if ok else 'unbuilt — visit to generate'}</td>"
            f"<td>{esc(name)}{esc(note)}</td>"
            f"<td><a href='~SRV~/{esc(d)}/'>visit →</a></td></tr>"
        )

    rows = "".join(row(d, reg[d], True) for d in built) + "".join(row(d, reg[d], False) for d in unbuilt)
    hosts = "\n".join(f"127.0.0.1 {d}" for d in sorted(reg)) or "# nothing here yet — build something first"
    loglines = "".join(f"<li>{esc(x.get('msg', ''))}</li>" for x in reversed(st.get("log", [])[-8:])) or "<li>nothing yet — build something</li>"

    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>.llm — mission control</title><style>{HUB_CSS}</style></head><body>
<div class='top'><span>THE .llm INTERNET</span><span>PRIVATE · FICTIONAL · {esc(PROVIDER).upper()} · GROWS ON CONTACT</span></div>
<div class='wrap'><h1>Mission control</h1>
<p class='sub'>Every .llm domain gets a unique design on first visit (one LLM call), and every article/comment page generates itself the first time you click it (one call each, then saved forever). Unvisited domains remember who linked to them, so the web knits together.</p>
<div class='stats'>
<div class='stat'><b>{len(built)}</b><span>sites built</span></div>
<div class='stat'><b>{len(unbuilt)}</b><span>doors not yet opened</span></div>
<div class='stat'><b>{n_pages}</b><span>pages on disk</span></div>
<div class='stat'><b>{safe_int(st.get('api_calls', 0))}</b><span>LLM calls</span></div>
</div>
<form method='get' action='/__build'><input name='domain' placeholder='somebrand.news.llm' required pattern='{BUILD_PATTERN}'><button>Build it</button></form>
<h2>The map</h2><table>{rows or '<tr><td colspan=4>empty — the internet is a blank page</td></tr>'}</table>
<h2>For /etc/hosts (optional — makes http://domain.llm work directly)</h2><pre>{esc(hosts)}</pre>
<h2>Recent activity</h2><ul>{loglines}</ul>
<p style='margin-top:34px;color:#5f9278;font-size:13px'>Everything generated here is fiction about fictional places. Revisits are served from disk — no extra LLM calls.</p>
</div></body></html>"""


def notfound_html(domain, rel, ddir):
    design = {}
    dj = ddir / "design.json"
    if dj.exists():
        try:
            design = json.loads(dj.read_text())
        except Exception:
            pass

    name = design.get("site_name", domain)
    accent = (design.get("palette") or {}).get("accent", "#35d07a")
    files = sorted(str(p.relative_to(ddir)) for p in ddir.rglob("*") if p.is_file() and p.suffix == ".html")
    links = "".join(f"<li><a href='~SRV~/{domain}/{esc(f)}'>{esc(f)}</a></li>" for f in files)

    return f"""<!doctype html><html><head><meta charset='utf-8'><title>404 — {esc(name)}</title>
<style>body{{font-family:ui-monospace,Menlo,monospace;background:#0b2e21;color:#eaf3ec;display:grid;place-items:center;min-height:100vh;margin:0}}.box{{max-width:560px;padding:0 20px}}h1{{font-size:3rem}}a{{color:{accent}}}li{{margin:6px 0}}</style></head>
<body><div class='box'><h1>404</h1>
<p><b>{esc(name)}</b> does not have <code>{esc(rel or '/')}</code>.<br>It has, however, these pages:</p>
<ul>{links}</ul>
<p><a href='~SRV~/{domain}/'>← home</a> · <a href='~SRV~/'>mission control</a></p></div></body></html>"""


def comments_disabled_html(domain, item, design):
    item = item or {}
    name = design.get("site_name", domain)
    accent = (design.get("palette") or {}).get("accent", "#35d07a")
    title = esc(item.get("title") or "this page")
    slug = item.get("slug") or ""
    back = f"~SRV~/{domain}/{slug}.html" if slug else f"~SRV~/{domain}/"

    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Comments disabled — {esc(name)}</title>
<style>
body{{font-family:ui-monospace,Menlo,monospace;background:#0b2e21;color:#eaf3ec;display:grid;place-items:center;min-height:100vh;margin:0}}
.box{{max-width:560px;padding:0 20px}}
h1{{font-size:2.4rem}}
a{{color:{accent}}}
.meta{{color:#9fc4ae;margin-top:12px}}
</style></head>
<body><div class='box'><h1>Comments disabled</h1>
<p><b>{title}</b> is an information-only page.</p>
<p>This site uses a product/information profile: no jokes, no snark, no trolling, and no user comment thread.</p>
<p><a href='{back}'>← back to page</a> · <a href='~SRV~/{domain}/'>home</a></p>
<p class='meta'>100% fiction · local .llm internet</p></div></body></html>"""


# ---------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    server_version = "LLMNet/3.0"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        try:
            self.route()
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, flush=True)

            safe_msg = str(e) or repr(e)
            self.send_html(
                500,
                f"""
                <h1>500</h1>
                <p>{esc(safe_msg)}</p>
                <pre>{esc(tb)}</pre>
                <p><a href='/'>back to mission control</a></p>
                """,
            )

    def _origin(self, host_hdr):
        if ORIGIN:
            return ORIGIN
        proto = self.headers.get("X-Forwarded-Proto") or "http"
        host = host_hdr or f"localhost:{PORT}"
        return f"{proto}://{host}"

    def route(self):
        host_hdr = self.headers.get("Host") or f"localhost:{PORT}"
        host = host_hdr.split(":")[0].lower()
        parts = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(parts.path)

        if path.endswith("/favicon.ico"):
            self.send_response(204)
            self.end_headers()
            return

        if path == "/__build":
            q = urllib.parse.parse_qs(parts.query)
            d = (q.get("domain", [""])[0]).lower().strip()
            d = fix_domain(d)
            if not d:
                self.send_html(400, "<h1>400</h1><p>That does not look like a <b>.llm</b> domain.</p>", host_hdr)
                return
            ensure_domain(d)
            self.send_response(302)
            self.send_header("Location", self._origin(host_hdr) + f"/{d}/")
            self.end_headers()
            return

        domain, rel = None, path
        if host.endswith(".llm") and DOMAIN_RE.match(host):
            domain, rel = host, path
        else:
            seg = path.strip("/").split("/", 1)
            if seg and DOMAIN_RE.match(seg[0].lower()):
                domain = seg[0].lower()
                rel = "/" + seg[1] if len(seg) > 1 else "/"

        if not domain:
            self.send_html(200, hub_html(), host_hdr)
            return

        ddir = SITES / domain
        if not (ddir / "design.json").exists() or not (ddir / "index.html").exists():
            ensure_domain(domain)

        self.serve_site(domain, ddir, rel, host_hdr)

    def safe_file(self, ddir, f):
        try:
            p = f.resolve()
            p.relative_to(ddir.resolve())
        except (OSError, ValueError):
            return False
        return p.is_file()

    def serve_site(self, domain, ddir, rel, host_hdr):
        rel = rel.strip("/")
        if rel in ("", "index.html"):
            f = ddir / "index.html"
            if self.safe_file(ddir, f):
                self.serve_file(f, host_hdr)
                return

        f = ddir / rel
        if self.safe_file(ddir, f):
            self.serve_file(f, host_hdr)
            return

        design = None
        dj = ddir / "design.json"
        if dj.exists():
            try:
                design = json.loads(dj.read_text())
            except Exception:
                pass

        kind = classify(rel, design) if design else None
        if kind:
            k, item = kind
            if k == "comments_disabled":
                self.send_html(200, comments_disabled_html(domain, item, design), host_hdr)
                return

            ensure_page(domain, rel, k, item)
            f = ddir / rel
            if self.safe_file(ddir, f):
                self.serve_file(f, host_hdr)
                return

        self.send_html(404, notfound_html(domain, rel, ddir), host_hdr)

    def serve_file(self, p, host_hdr):
        data = p.read_bytes().decode("utf-8", "replace")
        is_html = p.suffix == ".html"
        if is_html:
            origin = self._origin(host_hdr)
            data = data.replace("~SRV~", origin)

        out = data.encode("utf-8")
        ct = (
            "text/html; charset=utf-8"
            if is_html
            else ("application/json; charset=utf-8" if p.suffix == ".json" else "application/octet-stream")
        )

        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(out)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(out)

    def send_html(self, code, html, host_hdr=None):
        origin = self._origin(host_hdr)
        out = html.replace("~SRV~", origin).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


# ---------------------------------------------------------------- cli


SEED = [
    "apple.com.llm",
    "fark.com.llm",
    "weather.com.llm",
    "quantumtoes.blog.llm",
    "thebarkreport.news.llm",
    "sockpuppet.shop.llm",
    "capywiki.wiki.llm",
    "anonboard.forum.llm",
]


def main():
    global PROVIDER, MODEL, BASE, PORT, BIND, TEMPERATURE

    ap = argparse.ArgumentParser(description="llmnet — a fake .llm internet, generated with an LLM and saved to disk.")
    ap.add_argument("cmd", nargs="?", default="serve", choices=["serve", "build", "map"])
    ap.add_argument("domains", nargs="*")
    ap.add_argument("--seed", action="store_true", help="build the starter set of domains")
    ap.add_argument("--all", action="store_true", help="build every referenced-but-unbuilt domain (front pages only)")
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild specified domains")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--provider", default=PROVIDER, help="anthropic, openai, or ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--bind", default=BIND)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = ap.parse_args()

    PROVIDER = args.provider.lower()
    if PROVIDER not in DEFAULT_BASE:
        PROVIDER = "anthropic"

    BASE = (args.base_url or DEFAULT_BASE[PROVIDER]).rstrip("/")
    MODEL = args.model or {
        "anthropic": "claude-haiku-4-5",
        "openai": "local-model",
        "ollama": "llama3.1",
    }[PROVIDER]
    BIND = args.bind or "127.0.0.1"
    PORT = args.port
    TEMPERATURE = args.temperature

    ROOT.mkdir(parents=True, exist_ok=True)
    SITES.mkdir(parents=True, exist_ok=True)

    if args.cmd == "build":
        if args.all:
            reg = load_reg()
            targets = [d for d, v in sorted(reg.items()) if v.get("status") != "built"]
        elif args.seed:
            targets = SEED
        else:
            targets = args.domains

        if not targets:
            print("nothing to build. Try:  llmnet.py build apple.com.llm   (or --seed / --all)")
            return

        for d in targets:
            d = fix_domain(d)
            if not d:
                print(f"  ✗ invalid domain: {d}")
                continue

            if args.rebuild:
                dd = SITES / d
                if dd.exists():
                    shutil.rmtree(dd)

            try:
                ddir, created = ensure_domain(d)
                print(f"  ✓ {d} → front page ready at {ddir}/index.html" + ("" if created else "  (already built)"))
            except Exception as e:
                print(f"  ✗ {d}: {e}")

    elif args.cmd == "map":
        reg, st = load_reg(), load_state()
        if not reg:
            print("The .llm internet is empty. Build something first:  llmnet.py build apple.com.llm")
            return

        print(f"{'DOMAIN':40} {'STATUS':10} NOTED BY")
        for d in sorted(reg):
            v = reg[d]
            n_pages = len(list((SITES / d).rglob("*.html"))) if (SITES / d).exists() else 0
            refs = ", ".join(r.get("domain", "") for r in v.get("referenced_by", [])[:2])
            print(f"{d:40} {v.get('status','?'):10} {refs}   [{n_pages} pages generated]")

        print(f"\n{safe_int(st.get('api_calls', 0))} LLM calls · {safe_int(st.get('tokens_out', 0))} out tokens · root: {ROOT}")
        print("\n--- for /etc/hosts ---")
        for d in sorted(reg):
            print(f"127.0.0.1 {d}")

    else:
        key_status = "yes" if API_KEY else "no (local providers can work without a key)"
        print(f"""
  ══ .llm INTERNET v3 ══════════════════════════════════
  mission control    http://localhost:{args.port}/
  apple (product)    http://localhost:{args.port}/apple.com.llm/
  fark (snarky)      http://localhost:{args.port}/fark.com.llm/
  weather (Denver)   http://localhost:{args.port}/weather.com.llm/

  provider           {PROVIDER}
  model              {MODEL}
  base url           {BASE}
  api key set        {key_status}

  first visit to a domain      → 2 LLM calls (design metadata + front page HTML)
  first click on an article    → 1 LLM call (saved forever after)
  first click on its comments  → 1 LLM call (unique to that article, saved)
  everything else              → served from disk, free.

  root: {ROOT}
  ctrl-c to quit
""")
        httpd = ThreadingHTTPServer((BIND, args.port), Handler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye — the fake internet remains, saved at", ROOT)


if __name__ == "__main__":
    main()