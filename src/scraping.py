from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
import pdfplumber
import fitz

from .config import DEFAULT_USER_AGENT, RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)


def cache_path_for_url(url: str, suffix: str = "") -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    safe_suffix = suffix if suffix.startswith(".") or not suffix else f".{suffix.lstrip('.') }"
    return RAW_DIR / f"{digest}{safe_suffix}"


def download_url(url: str, *, suffix: str = "", force: bool = False, delay_seconds: float = 1.0) -> Path:
    cache_path = cache_path_for_url(url, suffix=suffix)
    if cache_path.exists() and not force:
        return cache_path

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    if delay_seconds:
        time.sleep(delay_seconds)
    return cache_path


def download_text_page(url: str, *, force: bool = False) -> Path:
    return download_url(url, suffix=".html", force=force)


def download_pdf(url: str, *, force: bool = False) -> Path:
    return download_url(url, suffix=".pdf", force=force)


def read_pdf_text(pdf_path: Path) -> str:
    chunks: list[str] = []
    with pdfplumber.open(pdf_path) as pdf_file:
        for page in pdf_file.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text)
    return "\n\n".join(chunks)


def read_pdf_tables(pdf_path: Path) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    with pdfplumber.open(pdf_path) as pdf_file:
        for page in pdf_file.pages:
            page_tables = page.extract_tables() or []
            for table in page_tables:
                tables.append(table)
    return tables


def read_pdf_text_with_fitz(pdf_path: Path) -> str:
    document = fitz.open(pdf_path)
    return "\n\n".join(page.get_text("text") for page in document)


def read_html_text(html_path: Path) -> str:
    return html_path.read_text(encoding="utf-8", errors="ignore")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def download_many(urls: Iterable[str], *, suffix: str = "", force: bool = False) -> list[Path]:
    downloaded_paths: list[Path] = []
    for url in urls:
        downloaded_paths.append(download_url(url, suffix=suffix, force=force))
    return downloaded_paths
