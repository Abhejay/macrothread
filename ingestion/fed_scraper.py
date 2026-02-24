"""
fed_scraper.py
--------------
Scrapes speeches and FOMC minutes from federalreserve.gov
and saves them locally with clean metadata.

Usage:
    python fed_scraper.py --type speeches --limit 20
    python fed_scraper.py --type minutes --limit 10
    python fed_scraper.py --type all --limit 20 --years 2023 2024 2025
"""

import re
import time
import json
import argparse
import requests
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup

BASE_URL = "https://www.federalreserve.gov"
MINUTES_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"

def speeches_url(year: int) -> str:
    return f"{BASE_URL}/newsevents/speech/{year}-speeches.htm"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "fed"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "MacroThread-Bot/1.0 (open-source economics research tool)"
}

REQUEST_DELAY = 1.5

def get_page(url: str) -> BeautifulSoup | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [error] Failed to fetch {url}: {e}")
        return None


def save_document(content: str, metadata: dict, filename: str) -> Path:
    doc_dir = RAW_DIR / filename
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "content.txt").write_text(content, encoding="utf-8")
    (doc_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return doc_dir


def already_downloaded(filename: str) -> bool:
    return (RAW_DIR / filename / "content.txt").exists()


def extract_date_from_url(url: str) -> str:
    """Pull 8-digit date from URLs like powell20241107a.htm"""
    match = re.search(r'(\d{8})', url)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return match.group(1)
    return ""

def parse_speech_links(years: list[int], limit: int) -> list[dict]:
    """
    The Fed organizes speeches by year at:
    federalreserve.gov/newsevents/speech/2025-speeches.htm
    Each page lists all speeches for that year with links.
    """
    speeches = []

    for year in years:
        if len(speeches) >= limit:
            break

        url = speeches_url(year)
        print(f"  Fetching {year} speeches: {url}")
        soup = get_page(url)
        if not soup:
            continue

        time.sleep(REQUEST_DELAY)

        # Find all links pointing to individual speech pages
        # Speech URLs follow the pattern: /newsevents/speech/lastname{date}{letter}.htm
        links = soup.find_all(
            "a",
            href=lambda h: h and "/newsevents/speech/" in h and h.endswith(".htm")
            and re.search(r'\d{8}', h)  # has a date in the URL
        )

        for link in links:
            if len(speeches) >= limit:
                break

            href = link.get("href", "")
            full_url = BASE_URL + href if href.startswith("/") else href
            title = link.get_text(strip=True)

            # Skip empty or navigation links
            if not title or len(title) < 5:
                continue

            date = extract_date_from_url(href)

            # Try to find speaker — often in a nearby <p> or <div>
            parent = link.find_parent("div") or link.find_parent("td")
            speaker = "Federal Reserve Official"
            if parent:
                # Look for text that contains "Governor", "Chair", "Vice Chair", "President"
                text = parent.get_text(" ", strip=True)
                speaker_match = re.search(
                    r'(Chair|Governor|Vice Chair|President)[^·\n]+', text
                )
                if speaker_match:
                    speaker = speaker_match.group(0).strip()

            speeches.append({
                "title": title,
                "speaker": speaker,
                "date": date,
                "url": full_url
            })

    print(f"  Found {len(speeches)} speeches")
    return speeches[:limit]


def scrape_speech_content(speech: dict) -> str | None:
    soup = get_page(speech["url"])
    if not soup:
        return None

    # Try multiple content containers — Fed pages vary slightly
    article = (
        soup.select_one("div#article") or
        soup.select_one("div.col-xs-12.col-sm-8.col-md-8") or
        soup.select_one("div#content") or
        soup.select_one("main")
    )

    if not article:
        print(f"    [warning] Could not find article body at {speech['url']}")
        return None

    # Remove clutter
    for tag in article.select("div.footnotes, nav, div#nav, div.sidebar, script, style"):
        tag.decompose()

    text = article.get_text(separator="\n", strip=True)

    if len(text) < 200:
        print(f"    [warning] Content too short ({len(text)} chars), skipping")
        return None

    return text


def scrape_speeches(limit: int = 20, years: list[int] = None):
    if years is None:
        current_year = datetime.now().year
        years = [current_year, current_year - 1]

    print(f"\n── Scraping Fed Speeches (years: {years}) ──────────────────")
    speeches = parse_speech_links(years=years, limit=limit)

    if not speeches:
        print("  [error] No speeches found.")
        print(f"  Verify this URL works in your browser:")
        print(f"  {speeches_url(years[0])}")
        return

    saved = 0
    skipped = 0

    for i, speech in enumerate(speeches, 1):
        url_slug = re.sub(r'[^a-z0-9]', '_', speech["url"].split("/")[-1].replace(".htm", "").lower())
        filename = f"{speech['date']}_{url_slug}" if speech['date'] else url_slug

        if already_downloaded(filename):
            print(f"  [{i}/{len(speeches)}] Skipping: {filename}")
            skipped += 1
            continue

        print(f"  [{i}/{len(speeches)}] {speech['title'][:60]}...")

        content = scrape_speech_content(speech)
        if not content:
            continue

        metadata = {
            "title": speech["title"],
            "speaker": speech["speaker"],
            "date": speech["date"],
            "source_url": speech["url"],
            "institution": "Federal Reserve",
            "country": "US",
            "region": "North America",
            "doc_type": "speech",
            "scraped_at": datetime.utcnow().isoformat()
        }

        doc_dir = save_document(content, metadata, filename)
        print(f"    ✓ {len(content):,} chars → {doc_dir.name}")
        saved += 1
        time.sleep(REQUEST_DELAY)

    print(f"\n  Done. Saved: {saved} | Skipped: {skipped} | Total: {len(speeches)}")

def parse_minutes_links(limit: int) -> list[dict]:
    print(f"\nFetching FOMC calendar: {MINUTES_URL}")
    soup = get_page(MINUTES_URL)
    if not soup:
        return []

    minutes = []
    links = soup.find_all("a", href=lambda h: h and "fomcminutes" in h)

    for link in links[:limit]:
        href = link.get("href", "")
        full_url = BASE_URL + href if href.startswith("/") else href
        date = extract_date_from_url(href)

        minutes.append({
            "title": f"FOMC Minutes {date}",
            "speaker": "Federal Open Market Committee",
            "date": date,
            "url": full_url
        })

    print(f"  Found {len(minutes)} minutes documents")
    return minutes


def scrape_minutes_content(minutes: dict) -> str | None:
    soup = get_page(minutes["url"])
    if not soup:
        return None

    article = (
        soup.select_one("div#article") or
        soup.select_one("div.col-xs-12.col-sm-8") or
        soup.select_one("main")
    )

    if not article:
        print(f"  [warning] Could not find content at {minutes['url']}")
        return None

    return article.get_text(separator="\n", strip=True)


def scrape_minutes(limit: int = 10):
    print("\n── Scraping FOMC Minutes ──────────────────────────")
    minutes_list = parse_minutes_links(limit)

    saved = 0
    skipped = 0

    for i, minutes in enumerate(minutes_list, 1):
        filename = f"{minutes['date']}_fomc_minutes"

        if already_downloaded(filename):
            print(f"  [{i}/{len(minutes_list)}] Skipping: {filename}")
            skipped += 1
            continue

        print(f"  [{i}/{len(minutes_list)}] {minutes['title']}...")

        content = scrape_minutes_content(minutes)
        if not content:
            continue

        metadata = {
            "title": minutes["title"],
            "speaker": "Federal Open Market Committee",
            "date": minutes["date"],
            "source_url": minutes["url"],
            "institution": "Federal Reserve",
            "country": "US",
            "region": "North America",
            "doc_type": "minutes",
            "scraped_at": datetime.utcnow().isoformat()
        }

        doc_dir = save_document(content, metadata, filename)
        print(f"    ✓ Saved → {doc_dir.name}")
        saved += 1
        time.sleep(REQUEST_DELAY)

    print(f"\n  Done. Saved: {saved} | Skipped: {skipped} | Total: {len(minutes_list)}")

def main():
    parser = argparse.ArgumentParser(description="Scrape Federal Reserve documents")
    parser.add_argument("--type", choices=["speeches", "minutes", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Years to scrape e.g. --years 2023 2024 2025")
    args = parser.parse_args()

    print(f"\nMacroThread Fed Scraper")
    print(f"Saving to: {RAW_DIR}\n")

    if args.type in ("speeches", "all"):
        scrape_speeches(limit=args.limit, years=args.years)

    if args.type in ("minutes", "all"):
        scrape_minutes(limit=args.limit)

    print("\n✓ Scraping complete.")


if __name__ == "__main__":
    main()