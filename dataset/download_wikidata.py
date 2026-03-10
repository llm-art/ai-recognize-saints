#!/usr/bin/env python3
"""
Wikidata Image Downloader

Downloads paintings from Wikidata that have Iconclass codes starting with '11H'
(Christian saints), filters to the top N iconclasses, and saves the images
along with test and ground truth files.

Speed notes:
  - Wikimedia serves full-resolution scans (often 10–50 MB each). Pass --max-width
    to request server-side thumbnails instead (e.g. --max-width 1024), which
    reduces transfer size by ~10x with no local resize needed.
  - Use --workers to download in parallel (default: 4). Keep it ≤ 8 to stay
    within Wikimedia's polite-use guidelines.

Usage:
    python download_wikidata.py
    python download_wikidata.py --max-width 800 --workers 4
    python download_wikidata.py --top-n 10 --delay 0.5 --force-redownload
"""

import argparse
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse, quote, unquote

import numpy as np
import pandas as pd
import requests
from PIL import Image
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 178956970

SPARQL_QUERY = """
SELECT ?painting ?image ?iconclass WHERE {
  ?painting wdt:P31 wd:Q3305213;
            wdt:P1257 ?iconclass.
  ?painting wdt:P18 ?image.
  FILTER(strstarts(?iconclass, '11H'))
}
"""

HEADERS = {
    'User-Agent': 'SaintRecognitionResearch/1.0 (research bot; contact via GitHub)'
}


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def query_wikidata() -> pd.DataFrame:
    """Run SPARQL query and return results as a DataFrame."""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(SPARQL_QUERY)
    sparql.setReturnFormat(JSON)
    sparql.setRequestMethod('GET')

    results = sparql.query().convert()
    rows = [
        {
            'painting': item['painting']['value'],
            'image': item['image']['value'],
            'iconclass': item['iconclass']['value'],
        }
        for item in results['results']['bindings']
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset='painting')
    logging.info(f"Fetched {len(df)} paintings from Wikidata")
    return df


def filter_top_iconclasses(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.Series]:
    """Strip sub-codes, keep top N iconclasses, and deduplicate by image URL."""
    df = df.copy()
    df['iconclass'] = df['iconclass'].str.extract(r'([^\)]+\))')
    counts = df['iconclass'].value_counts().head(top_n)
    filtered = df[df['iconclass'].isin(counts.index)].drop_duplicates(subset='image')
    logging.info(f"Filtered to {len(filtered)} images across top {top_n} iconclasses")
    return filtered, counts


def _extract_wikimedia_filename(url: str) -> str | None:
    """
    Extract the normalised Commons filename from any Wikimedia URL variant:
      - https://commons.wikimedia.org/wiki/Special:FilePath/File.jpg
      - http://commons.wikimedia.org/wiki/Special:FilePath/File.jpg
      - https://upload.wikimedia.org/wikipedia/commons[/thumb]/a/ab/File.jpg[/Npx-File.jpg]
    Returns the filename with spaces replaced by underscores, or None if unrecognised.
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)

    if 'Special:FilePath/' in path:
        filename = path.split('Special:FilePath/')[-1].split('?')[0]
    elif '/wikipedia/commons/' in path:
        parts = [p for p in path.split('/') if p]
        # thumb URLs end with  Npx-filename  — strip that synthetic segment
        last = parts[-1]
        if re.match(r'^\d+px-', last):
            last = re.sub(r'^\d+px-', '', last)
        filename = last
    else:
        return None

    return filename.replace(' ', '_')


def thumbnail_url(url: str, max_width: int) -> str:
    """
    Build a direct Wikimedia CDN thumbnail URL without any redirect hops.

    Wikimedia stores thumbnails at:
      https://upload.wikimedia.org/wikipedia/commons/thumb/{a}/{ab}/{filename}/{W}px-{filename}
    where {a}/{ab} are the first 1/2 hex chars of MD5(filename).

    TIF and TIFF originals are served as JPEG thumbnails (Wikimedia convention).
    Falls back to the Special:FilePath?width= approach for unrecognised URL formats.
    """
    filename = _extract_wikimedia_filename(url)
    if not filename:
        # Unknown URL format – fall back to query-parameter redirect
        parsed = urlparse(url)
        qs = f"width={max_width}"
        return urlunparse(parsed._replace(query=qs))

    ext = os.path.splitext(filename)[1].lower()
    thumb_name = f"{max_width}px-{filename}"
    # Wikimedia converts TIF/TIFF thumbnails to JPEG
    if ext in ('.tif', '.tiff'):
        thumb_name += '.jpg'

    md5 = hashlib.md5(filename.encode('utf-8')).hexdigest()
    a, ab = md5[0], md5[:2]

    enc_file = quote(filename, safe='')
    enc_thumb = quote(thumb_name, safe='')
    return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{a}/{ab}/{enc_file}/{enc_thumb}"


def download_image(
    url: str,
    save_path: str,
    max_width: int = 0,
    delay: float = 0.0,
    max_retries: int = 3,
) -> tuple[bool, int, int, str, str]:
    """
    Download an image from Wikimedia.

    When max_width > 0 the direct CDN thumbnail URL is constructed via MD5-based
    path (no redirects needed).  When max_width == 0 the original URL is used as-is.

    Returns:
        (success, width, height, reason, final_url)
        final_url is the resolved URL (after any remaining CDN redirects).
    """
    fetch_url = thumbnail_url(url, max_width) if max_width else url
    last_reason = "unknown error"

    for attempt in range(max_retries + 1):
        try:
            logging.info(f"GET {fetch_url}" + (f" (retry {attempt})" if attempt else ""))
            response = requests.get(fetch_url, headers=HEADERS, stream=True, timeout=60, allow_redirects=True)
            final_url = response.url  # resolved after any redirects

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                last_reason = f"HTTP 429 rate limited (retry-after: {retry_after}s)"
                logging.warning(
                    f"{last_reason} — {fetch_url} "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
                response.close()
                time.sleep(retry_after)
                continue  # retry after sleeping

            if response.status_code != 200:
                reason = f"HTTP {response.status_code}"
                logging.error(f"{reason} — {fetch_url}")
                return False, 0, 0, reason, fetch_url

            img_bytes = io.BytesIO()
            for chunk in response.iter_content(8192):
                img_bytes.write(chunk)
            img_bytes.seek(0)

            with Image.open(img_bytes) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                width, height = img.size
                img.save(save_path, 'JPEG', quality=95)

            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                if final_url != fetch_url:
                    logging.info(f"Redirected: {fetch_url} → {final_url}")
                logging.info(f"Saved {save_path} ({width}x{height})")
                time.sleep(delay)
                return True, width, height, "", final_url

            last_reason = "saved file has zero size"
            logging.warning(f"{last_reason}: {save_path}")
            os.remove(save_path)
            return False, 0, 0, last_reason, fetch_url

        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            logging.error(f"Failed {fetch_url}: {reason}")
            time.sleep(delay)
            return False, 0, 0, reason, fetch_url

    return False, 0, 0, last_reason, fetch_url


def resize_existing_images(jpeg_images_dir: str, max_width: int, workers: int) -> list[tuple[int, int]]:
    """
    Resize already-downloaded images that exceed max_width, in-place, using PIL.
    Images already within the limit are left untouched.
    Returns the list of (width, height) for all images after the pass.
    """
    files = [
        os.path.join(jpeg_images_dir, f)
        for f in os.listdir(jpeg_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]

    sizes: list[tuple[int, int]] = []
    resized = 0
    lock = threading.Lock()

    def process(path):
        nonlocal resized
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w <= max_width and h <= max_width:
                    with lock:
                        sizes.append((w, h))
                    return
                # Downscale preserving aspect ratio
                img.thumbnail((max_width, max_width), Image.LANCZOS)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(path, 'JPEG', quality=95)
                new_w, new_h = img.size
            with lock:
                sizes.append((new_w, new_h))
                resized += 1
            logging.info(f"Resized {path}: {w}x{h} → {new_w}x{new_h}")
        except Exception as e:
            logging.warning(f"Could not resize {path}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, p) for p in files]
        with tqdm(total=len(files), desc="Resizing") as pbar:
            for f in as_completed(futures):
                f.result()
                pbar.update(1)

    print(f"Resized {resized}/{len(files)} images (others were already ≤ {max_width}px).")
    return sizes


def save_statistics(image_sizes: list, failed_count: int, out_path: str) -> None:
    if not image_sizes:
        return
    widths = [w for w, _ in image_sizes]
    heights = [h for _, h in image_sizes]
    lines = [
        "Image Statistics:",
        f"Width:  {np.mean(widths):.2f} ± {np.std(widths):.2f} pixels",
        f"Height: {np.mean(heights):.2f} ± {np.std(heights):.2f} pixels",
        "Target from paper: Width: 778.84 ± 198.74, Height: 669.36 ± 174.18",
        f"\nTotal images: {len(image_sizes)}",
        f"Min width: {min(widths)}, Max width: {max(widths)}",
        f"Min height: {min(heights)}, Max height: {max(heights)}",
        f"Failed downloads: {failed_count}",
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wikidata painting images for saint recognition")
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top iconclasses to keep (default: 10)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Polite delay between requests per worker, in seconds (default: 0.5)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel download threads (default: 4, max recommended: 8)')
    parser.add_argument('--max-width', type=int, default=800,
                        help='Request server-side thumbnail at this width in pixels '
                             '(default: 800, matching ArtDL resolution). '
                             'Pass 0 to disable and download original sizes.')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Max retries on HTTP 429 rate-limit responses (default: 3)')
    parser.add_argument('--force-redownload', action='store_true',
                        help='Re-download images that already exist locally')
    parser.add_argument('--resize-existing', action='store_true',
                        help='Resize already-downloaded images that exceed --max-width in-place, '
                             'then exit without querying Wikidata or downloading anything new.')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    wikidata_dir = os.path.join(base_dir, 'wikidata')
    wikidata_data_dir = os.path.join(base_dir, 'wikidata-data')
    jpeg_images_dir = os.path.join(wikidata_dir, 'JPEGImages')

    os.makedirs(wikidata_dir, exist_ok=True)
    os.makedirs(wikidata_data_dir, exist_ok=True)
    os.makedirs(jpeg_images_dir, exist_ok=True)

    setup_logging(os.path.join(base_dir, 'image_download.log'))

    # --- Resize-only mode: no network calls ---
    if args.resize_existing:
        if not args.max_width:
            print("--resize-existing requires --max-width > 0.")
            return
        os.makedirs(jpeg_images_dir, exist_ok=True)
        sizes = resize_existing_images(jpeg_images_dir, args.max_width, args.workers)
        save_statistics(
            sizes, 0,
            os.path.join(wikidata_data_dir, 'image_statistics.txt'),
        )
        return

    # --- Step 1: SPARQL query ---
    print("Querying Wikidata...")
    df = query_wikidata()
    df.to_csv(os.path.join(wikidata_data_dir, 'paintings.csv'), index=False, quotechar="'")
    print(f"Saved {len(df)} paintings to paintings.csv")

    # --- Step 2: Filter top iconclasses ---
    df_filtered, iconclass_counts = filter_top_iconclasses(df, args.top_n)
    print(f"\nTop {args.top_n} iconclasses:\n{iconclass_counts.to_string()}\n")
    df_filtered.to_csv(os.path.join(wikidata_data_dir, 'wikidata.csv'), index=False, quotechar="'")
    iconclass_counts.to_csv(os.path.join(wikidata_data_dir, 'pre_classes.csv'), header=True)
    print(f"Saved {len(df_filtered)} filtered paintings to wikidata.csv")

    # --- Step 3: Parallel download ---
    image_data: list[dict] = []
    image_sizes: list[tuple[int, int]] = []
    failed_downloads: list[dict] = []
    skipped_images: list[dict] = []
    lock = threading.Lock()

    all_rows = list(df_filtered.iterrows())

    # Pre-filter: collect already-downloaded images without touching the thread pool
    rows = []
    for idx, row in all_rows:
        filename = row['painting'].split('/')[-1] + '.jpg'
        save_path = os.path.join(jpeg_images_dir, filename)
        if not args.force_redownload and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            try:
                with Image.open(save_path) as img:
                    width, height = img.size
                skipped_images.append({
                    'painting': row['painting'],
                    'image': row['image'],
                    'iconclass': row['iconclass'],
                    'width': width,
                    'height': height,
                })
                image_sizes.append((width, height))
            except Exception:
                rows.append((idx, row))  # corrupt — re-download
        else:
            rows.append((idx, row))

    print(f"Pre-check: {len(skipped_images)} already downloaded, {len(rows)} to fetch.")

    def process_row(args_row):
        idx, row = args_row
        filename = row['painting'].split('/')[-1] + '.jpg'
        save_path = os.path.join(jpeg_images_dir, filename)

        success, width, height, reason, final_url = download_image(
            row['image'],
            save_path,
            max_width=args.max_width,
            delay=args.delay,
            max_retries=args.max_retries,
        )

        with lock:
            if success:
                image_sizes.append((width, height))
                image_data.append({
                    'painting': row['painting'],
                    'image': final_url,
                    'iconclass': row['iconclass'],
                    'width': width,
                    'height': height,
                })
            else:
                failed_downloads.append({
                    'painting': row['painting'],
                    'image': row['image'],
                    'iconclass': row['iconclass'],
                    'reason': reason,
                })

            # Save progress every 50 completed items
            total_done = len(image_data) + len(skipped_images) + len(failed_downloads)
            if total_done % 50 == 0:
                with open(os.path.join(wikidata_data_dir, 'wikidata.json'), 'w') as f:
                    json.dump(image_data, f)

    mode = f"max-width={args.max_width}px" if args.max_width else "full-res (no cap)"
    print(f"Downloading {len(rows)} images ({mode}, {args.workers} workers)...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_row, r): r for r in rows}
        with tqdm(total=len(rows), desc="Downloading") as pbar:
            for future in as_completed(futures):
                future.result()  # re-raise any exception
                pbar.update(1)

    # Final saves
    with open(os.path.join(wikidata_data_dir, 'wikidata.json'), 'w') as f:
        json.dump(image_data, f)
    with open(os.path.join(wikidata_data_dir, 'failed_downloads.json'), 'w') as f:
        json.dump(failed_downloads, f)
    with open(os.path.join(wikidata_data_dir, 'skipped_images.json'), 'w') as f:
        json.dump(skipped_images, f)

    print(f"\nDownload complete.")
    print(f"  Downloaded:  {len(image_data)}")
    print(f"  Skipped:     {len(skipped_images)}")
    print(f"  Failed:      {len(failed_downloads)}")

    # --- Step 4: Statistics ---
    save_statistics(
        image_sizes,
        len(failed_downloads),
        os.path.join(wikidata_data_dir, 'image_statistics.txt'),
    )

    # --- Step 5: Test and ground truth files ---
    test_images = []
    ground_truth = []

    for item in image_data + skipped_images:
        image_filename = item['painting'].replace('http://www.wikidata.org/entity/', '')
        image_path = os.path.join(jpeg_images_dir, f'{image_filename}.jpg')
        if os.path.exists(image_path):
            test_images.append(image_filename)
            ground_truth.append({
                'item': image_filename,
                'class': item['iconclass'],
                'width': item.get('width', 0),
                'height': item.get('height', 0),
            })

    with open(os.path.join(wikidata_data_dir, '2_test.txt'), 'w') as f:
        f.write('\n'.join(test_images) + '\n')
    with open(os.path.join(wikidata_data_dir, '2_ground_truth.json'), 'w') as f:
        json.dump(ground_truth, f)

    print(f"\n2_test.txt and 2_ground_truth.json created with {len(test_images)} images.")
    logging.info(f"Done. Images: {len(test_images)}, failed: {len(failed_downloads)}")


if __name__ == '__main__':
    main()
