"""
push_to_qdrant.py — Push du lieu tu file JSONL output vao Qdrant

Chay:
    cd duongspa-ai-sale-train-pipeline

    # Push file moi nhat tu output/
    python push_to_qdrant.py

    # Push file cu the
    python push_to_qdrant.py --file output/qdrant_points_20260423_120000.jsonl

    # Xem truoc khong push (dry-run)
    python push_to_qdrant.py --dry-run

    # Khong recreate collection (giu nguyen du lieu cu, chi upsert them)
    python push_to_qdrant.py --no-recreate

    # Quet toan bo output/ va push tat ca file JSONL
    python push_to_qdrant.py --all
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

# Them thu muc goc vao path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from config import COLLECTION_NAME, OUTPUT_DIR, QDRANT_URL, VECTOR_SIZE
from qdrant_ops import client, create_collection, _check_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
DELAY_BETWEEN_BATCHES = float(os.environ.get("QDRANT_PUSH_DELAY", "0.1"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def find_latest_jsonl(output_dir: str) -> str | None:
    """Tim file qdrant_points_*.jsonl moi nhat trong output/."""
    pattern = os.path.join(output_dir, "qdrant_points_*.jsonl")
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def find_all_jsonl(output_dir: str) -> list[str]:
    """Liet ke tat ca file qdrant_points_*.jsonl trong output/."""
    pattern = os.path.join(output_dir, "qdrant_points_*.jsonl")
    return sorted(glob.glob(pattern), reverse=True)


def load_jsonl(filepath: str) -> list[dict]:
    """Doc file JSONL, tra ve list points."""
    points = []
    errors = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                # Validate co du truong can thiet
                if "id" not in p or "vector" not in p or "payload" not in p:
                    logger.warning(f"  Line {lineno}: thieu truong id/vector/payload, bo qua")
                    errors += 1
                    continue
                if not isinstance(p["vector"], list) or len(p["vector"]) == 0:
                    logger.warning(f"  Line {lineno}: vector rong, bo qua")
                    errors += 1
                    continue
                points.append(p)
            except json.JSONDecodeError as e:
                logger.warning(f"  Line {lineno}: JSON parse error: {e}")
                errors += 1
    if errors:
        logger.warning(f"  Bo qua {errors} dong loi")
    return points


def validate_vector_size(points: list[dict]) -> int:
    """Kiem tra vector size cua file va tra ve gia tri."""
    if not points:
        return 0
    sizes = set(len(p["vector"]) for p in points[:10])
    if len(sizes) > 1:
        raise ValueError(f"Vector size khong dong nhat trong file: {sizes}")
    actual = list(sizes)[0]
    if actual != VECTOR_SIZE:
        print(f"\n  [WARN] Vector size trong file={actual} != VECTOR_SIZE config={VECTOR_SIZE}")
        print(f"  Kiem tra lai .env: VECTOR_SIZE={actual}")
    return actual


def push_points(points: list[dict], dry_run: bool = False) -> int:
    """Push points vao Qdrant. Tra ve so points da push thanh cong."""
    from qdrant_client.models import PointStruct

    total = len(points)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    pushed = 0

    for batch_idx, i in enumerate(range(0, total, BATCH_SIZE), 1):
        batch_raw = points[i:i + BATCH_SIZE]
        batch = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in batch_raw
        ]

        if dry_run:
            print(f"  [DRY-RUN] Batch {batch_idx}/{total_batches}: {len(batch)} points (khong push)")
            pushed += len(batch)
            continue

        try:
            result = client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,
            )
            pushed += len(batch)
            status = getattr(result, "status", "ok")
            print(f"  Batch {batch_idx}/{total_batches}: {len(batch)} points | status={status} | total={pushed}/{total}")
        except Exception as e:
            logger.error(f"  [ERROR] Batch {batch_idx}: {e}")
            raise

        if DELAY_BETWEEN_BATCHES > 0 and i + BATCH_SIZE < total:
            time.sleep(DELAY_BETWEEN_BATCHES)

    return pushed


def process_file(filepath: str, recreate: bool, dry_run: bool) -> bool:
    """Xu ly mot file JSONL. Tra ve True neu thanh cong."""
    sep(f"File: {os.path.basename(filepath)}")
    print(f"  Path: {filepath}")

    # 1. Doc file
    print("  Dang doc file...")
    points = load_jsonl(filepath)
    if not points:
        print("  [ERROR] File rong hoac khong co point hop le!")
        return False

    print(f"  Points doc duoc  : {len(points)}")

    # 2. Validate vector
    try:
        vec_size = validate_vector_size(points)
        print(f"  Vector size      : {vec_size}")
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return False

    # 3. Kiem tra ket noi Qdrant
    print(f"  Qdrant URL       : {QDRANT_URL}")
    print(f"  Collection       : {COLLECTION_NAME}")
    try:
        _check_connection()
        print("  Ket noi Qdrant   : OK")
    except ConnectionError as e:
        print(f"  [ERROR] {e}")
        return False

    # 4. Recreate collection neu can
    if recreate and not dry_run:
        print("  Dang recreate collection...")
        create_collection()
    elif recreate and dry_run:
        print("  [DRY-RUN] Se recreate collection")
    else:
        # Kiem tra collection ton tai chua
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            print(f"  Collection chua ton tai -> tu dong tao moi...")
            if not dry_run:
                create_collection()
            else:
                print("  [DRY-RUN] Se tao collection moi")
        else:
            count_before = client.count(COLLECTION_NAME, exact=True).count
            print(f"  Collection hien co: {count_before} points (se upsert them, khong xoa)")

    # 5. Push
    print(f"\n  Bat dau push {len(points)} points (batch={BATCH_SIZE})...")
    start = time.time()
    pushed = push_points(points, dry_run=dry_run)
    elapsed = time.time() - start

    if dry_run:
        print(f"\n  [DRY-RUN] Xong. Se push {pushed} points (chua thuc su push)")
        return True

    # 6. Verify
    count_after = client.count(COLLECTION_NAME, exact=True).count
    print(f"\n  Pushed           : {pushed} points trong {elapsed:.1f}s")
    print(f"  Verified (count) : {count_after} points trong collection")

    if pushed == len(points):
        print("  [OK] Push thanh cong!")
        return True
    else:
        print(f"  [WARN] Pushed {pushed}/{len(points)}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Push du lieu JSONL tu output/ vao Qdrant"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Duong dan file JSONL cu the. Mac dinh: file moi nhat trong output/",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Push tat ca file qdrant_points_*.jsonl trong output/",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Khong xoa collection cu, chi upsert them (mac dinh: recreate)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Xem truoc: doc file va validate nhung khong push vao Qdrant",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Thu muc chua file JSONL (mac dinh: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DuongSpa — Push JSONL to Qdrant")
    print("=" * 60)
    print(f"  Qdrant URL  : {QDRANT_URL}")
    print(f"  Collection  : {COLLECTION_NAME}")
    print(f"  VECTOR_SIZE : {VECTOR_SIZE}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Recreate    : {not args.no_recreate}")
    print(f"  Dry-run     : {args.dry_run}")
    if args.dry_run:
        print("\n  [DRY-RUN MODE] Khong co gi duoc ghi vao Qdrant!")

    recreate = not args.no_recreate

    # Chon file(s) de push
    if args.file:
        files = [args.file]
    elif args.all:
        files = find_all_jsonl(args.output_dir)
        if not files:
            print(f"\n[ERROR] Khong tim thay file qdrant_points_*.jsonl trong '{args.output_dir}'")
            sys.exit(1)
        print(f"\n  Tim thay {len(files)} file JSONL:")
        for f in files:
            size_kb = os.path.getsize(f) / 1024
            print(f"    - {os.path.basename(f)} ({size_kb:.0f} KB)")
        # Chi recreate lan dau (lan sau upsert them)
    else:
        latest = find_latest_jsonl(args.output_dir)
        if not latest:
            print(f"\n[ERROR] Khong tim thay file qdrant_points_*.jsonl trong '{args.output_dir}'")
            print(f"  Chay main.py truoc de tao du lieu, hoac dung --file de chi dinh file cu the")
            sys.exit(1)
        files = [latest]
        print(f"\n  File moi nhat: {os.path.basename(latest)}")
        size_kb = os.path.getsize(latest) / 1024
        print(f"  Size: {size_kb:.0f} KB")

    # Push tung file
    success_count = 0
    for idx, filepath in enumerate(files):
        # Neu push --all: chi recreate cho file dau, cac file sau upsert them
        do_recreate = recreate if idx == 0 else False
        ok = process_file(filepath, recreate=do_recreate, dry_run=args.dry_run)
        if ok:
            success_count += 1

    sep(f"Tong ket: {success_count}/{len(files)} files pushed thanh cong")
    sys.exit(0 if success_count == len(files) else 1)


if __name__ == "__main__":
    main()
