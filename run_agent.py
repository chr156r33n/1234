#!/usr/bin/env python3

import argparse
import logging
import os
from urllib.parse import urlparse
from typing import Tuple, List

import requests
from bs4 import BeautifulSoup

from mcp_client import SemrushMCPClient
from plp_pipeline import run_plp_pipeline  # or your agent pipeline


# -----------------------------------------------------
# 1. Hard-coded GCP project ID
# -----------------------------------------------------
GCP_PROJECT_ID = "marketingdata-393009"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)


def extract_domain(url: str) -> str:
    """
    Extract domain from a URL, no www-stripping magic unless you want that.
    """
    parsed = urlparse(url)
    return parsed.netloc.lower()


def fetch_page_content(url: str) -> Tuple[str, str, str]:
    logger.info(f"[Fetch] Fetching {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    title = (soup.title.string or "").strip() if soup.title else ""
    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(strip=True) if h1_tag else ""

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body_text = " ".join(soup.get_text(separator=" ").split())

    return title, h1, body_text


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "root"
    return path.replace("/", "_")[:80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PLP keyword research pipeline.")

    parser.add_argument(
        "--database",
        default=os.environ.get("SEMRUSH_DATABASE", "us"),
        help="Semrush database, e.g. 'us', 'uk'.",
    )

    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="PLP URL to analyse. Pass multiple times for multiple URLs.",
    )

    parser.add_argument(
        "--output",
        help="Output CSV path or directory.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("[Main] Debug logging enabled")

    urls: List[str] = args.url
    multi = len(urls) > 1

    logger.info(
        f"[Main] Running pipeline for {len(urls)} URL(s) | "
        f"project={GCP_PROJECT_ID} | db={args.database}"
    )

    # One MCP client shared across all runs
    semrush_mcp = SemrushMCPClient()

    for url in urls:
        # 2. Extract the target domain from the URL
        target_domain = extract_domain(url)
        logger.info(f"[Main] Derived target domain: {target_domain}")

        title, h1, body = fetch_page_content(url)

        # Determine output path
        if args.output and not multi:
            output_path = args.output
        else:
            slug = slug_from_url(url)
            output_basename = f"plp_keywords_{slug}.csv"

            if args.output and multi:
                output_path = os.path.join(args.output, output_basename)
            else:
                output_path = output_basename

        logger.info(f"[Main] Output file: {output_path}")

        run_plp_pipeline(
            project_id=GCP_PROJECT_ID,
            semrush_mcp=semrush_mcp,
            database=args.database,
            target_domain=target_domain,
            page_title=title,
            page_h1=h1,
            page_body=body,
            output_csv_path=output_path,
            debug=args.debug,
        )

    logger.info("[Main] All URLs processed.")


if __name__ == "__main__":
    main()
