#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


SOURCE_URL = "https://cert.pl/ouch/"
USER_AGENT = "ouch-dataset-scraper/1.0 (+https://cert.pl/ouch/)"


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def parse_issue_meta(link_text: str) -> tuple[str, str]:
    match = re.match(r"^\s*(\d{2}/\d{4})\s+(.+?)\s*$", link_text)
    if not match:
        return "", link_text.strip()
    return match.group(1), match.group(2)


def fetch_issue_links(session: requests.Session, source_url: str) -> list[dict]:
    response = session.get(source_url, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    issues = []

    for anchor in soup.select("a[href]"):
        text = anchor.get_text(" ", strip=True)
        href = anchor.get("href", "").strip()
        if not text or not href:
            continue
        if not re.match(r"^\d{2}/\d{4}\s+", text):
            continue

        issue_date, title = parse_issue_meta(text)
        absolute_link = urljoin(source_url, href)
        issues.append(
            {
                "issue_date": issue_date,
                "title": title,
                "source_url": absolute_link,
            }
        )

    return issues


def resolve_pdf_url(session: requests.Session, url: str) -> str:
    # Follow redirects and read final URL.
    response = session.get(url, timeout=45, allow_redirects=True, stream=True)
    response.raise_for_status()
    final_url = response.url
    content_type = response.headers.get("Content-Type", "").lower()
    response.close()

    # Some links end without .pdf but still serve PDF.
    if "pdf" in content_type or final_url.lower().endswith(".pdf"):
        return final_url
    return ""


def download_pdf(session: requests.Session, url: str, destination: Path) -> bool:
    response = session.get(url, timeout=90, stream=True)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        response.close()
        return False

    with destination.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                file_handle.write(chunk)
    response.close()
    return True


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text.strip())
    return "\n\n".join(pages_text).strip()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - max(0, overlap_chars))
    return chunks


def export_ml_formats(
    dataset: list[dict],
    output_dir: Path,
    chunk_chars: int,
    overlap_chars: int,
) -> None:
    processed_dir = output_dir / "processed"
    text_dir = processed_dir / "txt"
    processed_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    documents_jsonl = processed_dir / "documents_clean.jsonl"
    documents_csv = processed_dir / "documents_clean.csv"
    chunks_jsonl = processed_dir / "chunks.jsonl"

    chunk_count = 0
    doc_count = 0

    with (
        documents_jsonl.open("w", encoding="utf-8") as docs_jsonl_handle,
        chunks_jsonl.open("w", encoding="utf-8") as chunks_jsonl_handle,
        documents_csv.open("w", encoding="utf-8", newline="") as docs_csv_handle,
    ):
        csv_writer = csv.DictWriter(
            docs_csv_handle,
            fieldnames=[
                "doc_id",
                "issue_date",
                "title",
                "source_url",
                "pdf_url",
                "pdf_file",
                "status",
                "text_chars",
            ],
        )
        csv_writer.writeheader()

        for record in dataset:
            if record.get("status") not in {"ok", "empty_text"}:
                continue

            issue_date = record.get("issue_date", "")
            title = record.get("title", "")
            source_url = record.get("source_url", "")
            pdf_url = record.get("pdf_url", "")
            pdf_file = record.get("pdf_file", "")
            text = clean_text(record.get("text", ""))
            if not text:
                continue

            doc_id = safe_filename(f"{issue_date}_{title}") or safe_filename(title) or "doc"
            txt_path = text_dir / f"{doc_id}.txt"
            txt_path.write_text(text, encoding="utf-8")

            doc_record = {
                "doc_id": doc_id,
                "issue_date": issue_date,
                "title": title,
                "source_url": source_url,
                "pdf_url": pdf_url,
                "pdf_file": pdf_file,
                "status": record.get("status", ""),
                "text_chars": len(text),
                "text": text,
                "txt_file": str(txt_path.relative_to(output_dir)),
            }

            docs_jsonl_handle.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
            csv_writer.writerow(
                {
                    "doc_id": doc_id,
                    "issue_date": issue_date,
                    "title": title,
                    "source_url": source_url,
                    "pdf_url": pdf_url,
                    "pdf_file": pdf_file,
                    "status": record.get("status", ""),
                    "text_chars": len(text),
                }
            )

            chunks = split_into_chunks(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
            for idx, chunk_text in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{doc_id}_chunk_{idx:04d}",
                    "doc_id": doc_id,
                    "issue_date": issue_date,
                    "title": title,
                    "chunk_index": idx,
                    "chunk_chars": len(chunk_text),
                    "text": chunk_text,
                }
                chunks_jsonl_handle.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                chunk_count += 1

            doc_count += 1

    print(f"Saved cleaned docs JSONL: {documents_jsonl}")
    print(f"Saved cleaned docs CSV:   {documents_csv}")
    print(f"Saved chunks JSONL:       {chunks_jsonl}")
    print(f"Saved cleaned TXT files:  {text_dir}")
    print(f"ML export stats -> documents: {doc_count}, chunks: {chunk_count}")


def build_dataset(
    output_dir: Path,
    json_name: str,
    jsonl_name: str,
    sleep_seconds: float,
    chunk_chars: int,
    overlap_chars: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    issues = fetch_issue_links(session, SOURCE_URL)
    dataset = []

    for index, issue in enumerate(issues, start=1):
        source_url = issue["source_url"]
        issue_date = issue["issue_date"]
        title = issue["title"]

        print(f"[{index}/{len(issues)}] Processing: {issue_date} {title}")
        record = {
            "issue_date": issue_date,
            "title": title,
            "source_url": source_url,
            "pdf_url": "",
            "pdf_file": "",
            "text": "",
            "status": "skipped",
            "error": "",
        }

        try:
            pdf_url = resolve_pdf_url(session, source_url)
            if not pdf_url:
                record["status"] = "non_pdf_link"
                dataset.append(record)
                continue

            parsed = urlparse(pdf_url)
            original_name = Path(parsed.path).name or f"ouch_{issue_date.replace('/', '-')}.pdf"
            if not original_name.lower().endswith(".pdf"):
                original_name = f"{original_name}.pdf"
            local_name = safe_filename(f"{issue_date}_{original_name}")
            local_path = pdf_dir / local_name

            ok = download_pdf(session, pdf_url, local_path)
            if not ok:
                record["status"] = "download_not_pdf"
                dataset.append(record)
                continue

            text = extract_pdf_text(local_path)
            record["pdf_url"] = pdf_url
            record["pdf_file"] = str(local_path.relative_to(output_dir))
            record["text"] = text
            record["status"] = "ok" if text else "empty_text"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"

        dataset.append(record)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    json_path = output_dir / json_name
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)

    jsonl_path = output_dir / jsonl_name
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in dataset:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    ok_count = sum(1 for item in dataset if item["status"] == "ok")
    print(f"Saved dataset: {json_path}")
    print(f"Saved jsonl:  {jsonl_path}")
    print(f"Total issues: {len(dataset)} | successful text extractions: {ok_count}")
    export_ml_formats(
        dataset=dataset,
        output_dir=output_dir,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape CERT Polska OUCH! issues, download PDFs, and export JSON/JSONL dataset."
    )
    parser.add_argument("--output-dir", default="data/ouch_dataset", help="Output directory")
    parser.add_argument("--json-name", default="ouch_issues.json", help="JSON output file name")
    parser.add_argument("--jsonl-name", default="ouch_issues.jsonl", help="JSONL output file name")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between issues to reduce request burst",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1800,
        help="Target number of characters per training chunk",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=200,
        help="Character overlap between adjacent chunks",
    )
    args = parser.parse_args()

    build_dataset(
        output_dir=Path(args.output_dir),
        json_name=args.json_name,
        jsonl_name=args.jsonl_name,
        sleep_seconds=args.sleep_seconds,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )


if __name__ == "__main__":
    main()
