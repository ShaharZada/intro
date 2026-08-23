#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_photos.py — בורר תמונות לסרטון בת מצווה
===============================================

מקבל תיקייה עם תמונות (עדיף ייצוא Google Takeout), ומחזיר מבחר תמונות
מפוזר על פני כל שנות החיים, אחרי סינון טכני והסרת כפילויות.

מה הוא עושה:
  1. מוצא תאריך צילום אמיתי לכל תמונה (JSON של Takeout > EXIF > שם קובץ > mtime)
  2. מחשב חדות, חשיפה, רזולוציה, וזיהוי פנים
  3. מאתר כפילויות וסדרות בּרסט (perceptual hash) ומשאיר את הטובה בכל צביר
  4. מחלק מכסות לפי גיל, כך שכל שנה מיוצגת — ולא רק השנתיים האחרונות
  5. מעתיק את הנבחרות לתיקיית פלט ממוינת + מייצר דוח CSV

התקנה:
    pip install pillow numpy opencv-python pillow-heif tqdm

הרצה בסיסית:
    python select_photos.py --input ./Takeout/Google\\ Photos/אלבום \\
                            --output ./selected \\
                            --birthdate 2014-03-21 \\
                            --total 300

הרצה חוזרת עם פרמטרים אחרים מהירה מאוד — התוצאות מנותחות פעם אחת ונשמרות ב-cache.
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("חסרות ספריות. הרץ: pip install pillow numpy opencv-python pillow-heif tqdm")

try:
    import cv2
except ImportError:
    sys.exit("חסר opencv. הרץ: pip install opencv-python")

# תמיכה ב-HEIC (תמונות אייפון) — אופציונלי
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".m4v", ".3gp", ".mkv"}

CACHE_NAME = ".photo_analysis_cache.json"


# ---------------------------------------------------------------- תאריכים

def build_takeout_index(root: Path):
    """
    בונה מיפוי שם-קובץ -> תאריך צילום מתוך קובצי ה-JSON של Takeout.
    גוגל חותכת שמות ארוכים ומוסיפה סיומות משתנות, ולכן ממפים לפי שדה
    "title" שבתוך ה-JSON עצמו — זה אמין הרבה יותר משם קובץ ה-JSON.
    """
    index = {}
    json_files = list(root.rglob("*.json"))
    for jf in tqdm(json_files, desc="קורא metadata של Takeout"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        ts = None
        for key in ("photoTakenTime", "creationTime"):
            node = data.get(key)
            if isinstance(node, dict) and node.get("timestamp"):
                try:
                    ts = int(node["timestamp"])
                    break
                except (TypeError, ValueError):
                    pass
        if ts is None:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        title = data.get("title")
        if title:
            index[title.lower()] = dt
        # גיבוי: גם לפי שם קובץ ה-JSON עצמו
        stem = jf.name
        for suffix in (".supplemental-metadata.json", ".json"):
            if stem.endswith(suffix):
                index.setdefault(stem[: -len(suffix)].lower(), dt)
                break
    return index


FILENAME_DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})"),   # 20190313 / 2019-03-13
    re.compile(r"(19\d{2})[-_.]?(\d{2})[-_.]?(\d{2})"),
]


def date_from_filename(name: str):
    for pat in FILENAME_DATE_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    return datetime(y, mo, d)
            except ValueError:
                continue
    return None


def date_from_exif(img: Image.Image):
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    # 36867 = DateTimeOriginal, 36868 = DateTimeDigitized, 306 = DateTime
    for tag in (36867, 36868, 306):
        val = exif.get(tag)
        if val:
            try:
                return datetime.strptime(str(val).strip(), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue
    return None


def resolve_date(path: Path, img, takeout_index):
    """מחזיר (תאריך, מקור_התאריך). סדר עדיפויות מהאמין לפחות אמין."""
    dt = takeout_index.get(path.name.lower())
    if dt:
        return dt, "takeout"
    dt = date_from_exif(img) if img is not None else None
    if dt:
        return dt, "exif"
    dt = date_from_filename(path.name)
    if dt:
        return dt, "filename"
    return datetime.fromtimestamp(path.stat().st_mtime), "mtime"


# ---------------------------------------------------------------- ניתוח תמונה

_face_cascade = None
_profile_cascade = None


def get_cascades():
    global _face_cascade, _profile_cascade
    if _face_cascade is None:
        base = cv2.data.haarcascades
        _face_cascade = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        _profile_cascade = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
    return _face_cascade, _profile_cascade


def dhash(gray_small: np.ndarray) -> str:
    """Perceptual hash — 64 ביט. תמונות דומות מקבלות hash קרוב."""
    resized = cv2.resize(gray_small, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = np.packbits(diff.flatten())
    return bits.tobytes().hex()


def hamming(h1: str, h2: str) -> int:
    a = int(h1, 16)
    b = int(h2, 16)
    return bin(a ^ b).count("1")


def analyze_image(path: Path, takeout_index, detect_faces=True):
    """מנתח תמונה בודדת ומחזיר dict עם כל המדדים, או None אם נכשל."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)   # מיישר לפי אוריינטציה
    except Exception:
        return None

    dt, dt_source = resolve_date(path, img, takeout_index)
    width, height = img.size
    megapixels = (width * height) / 1_000_000

    try:
        rgb = img.convert("RGB")
    except Exception:
        return None

    # מקטין לניתוח — מהיר בהרבה, ולא פוגע במדדים
    work = rgb.copy()
    work.thumbnail((800, 800), Image.LANCZOS)
    arr = np.asarray(work)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # חדות: שונות של לפלסיאן. ערך נמוך = מטושטש
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # חשיפה
    mean_lum = float(gray.mean())
    contrast = float(gray.std())
    clipped_dark = float((gray < 8).mean())
    clipped_bright = float((gray > 247).mean())

    # פנים
    faces = []
    if detect_faces:
        frontal, profile = get_cascades()
        eq = cv2.equalizeHist(gray)
        det = frontal.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28))
        faces = list(det) if len(det) else []
        if not faces:
            det = profile.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28))
            faces = list(det) if len(det) else []

    img_area = gray.shape[0] * gray.shape[1]
    face_count = len(faces)
    largest_face_frac = 0.0
    if face_count:
        largest_face_frac = max(w * h for (_, _, w, h) in faces) / img_area

    return {
        "path": str(path),
        "name": path.name,
        "datetime": dt.isoformat(),
        "date_source": dt_source,
        "width": width,
        "height": height,
        "megapixels": round(megapixels, 2),
        "orientation": "portrait" if height > width else ("square" if height == width else "landscape"),
        "sharpness": round(sharpness, 2),
        "mean_lum": round(mean_lum, 1),
        "contrast": round(contrast, 2),
        "clipped_dark": round(clipped_dark, 4),
        "clipped_bright": round(clipped_bright, 4),
        "face_count": face_count,
        "largest_face_frac": round(largest_face_frac, 4),
        "dhash": dhash(gray),
    }


# ---------------------------------------------------------------- ניקוד

def percentile_rank(values):
    """ממיר רשימת ערכים לדירוג יחסי 0..1 — עמיד לחריגים."""
    arr = np.asarray(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr))
    return ranks / max(len(arr) - 1, 1)


def exposure_score(rec):
    """1.0 לחשיפה טובה, יורד ככל שהתמונה חשוכה/שרופה/שטוחה."""
    lum = rec["mean_lum"]
    # אופטימום סביב 118, ירידה לינארית לכיוון הקצוות
    s = 1.0 - abs(lum - 118) / 118
    s = max(s, 0.0)
    # קונטרסט נמוך = תמונה שטוחה
    if rec["contrast"] < 25:
        s *= 0.6
    # פיקסלים שרופים או שחורים לגמרי
    s *= max(0.0, 1.0 - rec["clipped_bright"] * 4)
    s *= max(0.0, 1.0 - rec["clipped_dark"] * 2)
    return max(0.0, min(1.0, s))


def face_score(rec, use_faces):
    if not use_faces:
        return 0.5
    n = rec["face_count"]
    if n == 0:
        return 0.0
    # פנים גדולות בפריים = פורטרט טוב. 12% מהפריים ומעלה נחשב מלא.
    size = min(rec["largest_face_frac"] / 0.12, 1.0)
    # מעדיף 1-4 אנשים; המון פנים = תמונה קבוצתית רחוקה
    count_factor = 1.0 if n <= 4 else 0.75
    return min(1.0, (0.45 + 0.55 * size) * count_factor)


def compute_scores(records, use_faces, w_sharp, w_face, w_expo, w_res):
    sharp_rank = percentile_rank([r["sharpness"] for r in records])
    res_rank = percentile_rank([r["megapixels"] for r in records])
    for i, r in enumerate(records):
        r["sharp_rank"] = float(sharp_rank[i])
        r["expo_score"] = exposure_score(r)
        r["face_score"] = face_score(r, use_faces)
        r["res_rank"] = float(res_rank[i])
        r["score"] = round(100 * (
            w_sharp * r["sharp_rank"]
            + w_face * r["face_score"]
            + w_expo * r["expo_score"]
            + w_res * r["res_rank"]
        ), 2)
    return records


# ---------------------------------------------------------------- כפילויות

def dedupe(records, hash_threshold=8, burst_seconds=12):
    """
    מקבץ תמונות דומות ומשאיר את בעלת הניקוד הגבוה בכל צביר.
    ההשוואה נעשית רק בתוך אותו יום — מונע O(n²) על כל האוסף.
    """
    by_day = defaultdict(list)
    for r in records:
        by_day[r["datetime"][:10]].append(r)

    kept, dropped = [], []
    for day, items in by_day.items():
        items.sort(key=lambda r: r["datetime"])
        clusters = []  # כל צביר: {"reps": [hash...], "members": [rec...], "last_ts": dt}
        for r in items:
            ts = datetime.fromisoformat(r["datetime"])
            placed = False
            for c in clusters:
                close_in_time = abs((ts - c["last_ts"]).total_seconds()) <= burst_seconds
                similar = any(hamming(r["dhash"], h) <= hash_threshold for h in c["reps"])
                if similar or (close_in_time and any(
                        hamming(r["dhash"], h) <= hash_threshold + 6 for h in c["reps"])):
                    c["members"].append(r)
                    c["reps"].append(r["dhash"])
                    c["last_ts"] = ts
                    placed = True
                    break
            if not placed:
                clusters.append({"reps": [r["dhash"]], "members": [r], "last_ts": ts})

        for c in clusters:
            best = max(c["members"], key=lambda r: r["score"])
            best["cluster_size"] = len(c["members"])
            kept.append(best)
            for m in c["members"]:
                if m is not best:
                    dropped.append(m)
    return kept, dropped


# ---------------------------------------------------------------- מכסות

def age_at(dt: datetime, birthdate: datetime) -> int:
    years = dt.year - birthdate.year
    if (dt.month, dt.day) < (birthdate.month, birthdate.day):
        years -= 1
    return years


def allocate_quotas(buckets, total, min_per_bucket):
    """
    מחלק את המכסה בין שנות הגיל: קודם רצפה מינימלית לכל שנה שיש בה
    מספיק תמונות, ואז את היתרה באופן יחסי לזמינות.
    """
    keys = sorted(buckets.keys())
    avail = {k: len(buckets[k]) for k in keys}
    quota = {k: min(min_per_bucket, avail[k]) for k in keys}

    remaining = total - sum(quota.values())
    # מחלק את היתרה בסבבים, מדלג על שנים שנגמרו להן התמונות
    while remaining > 0:
        hungry = [k for k in keys if quota[k] < avail[k]]
        if not hungry:
            break
        weights = {k: avail[k] - quota[k] for k in hungry}
        wsum = sum(weights.values())
        added = 0
        for k in hungry:
            take = max(1, int(round(remaining * weights[k] / wsum)))
            take = min(take, avail[k] - quota[k], remaining - added)
            quota[k] += take
            added += take
            if added >= remaining:
                break
        if added == 0:
            break
        remaining -= added
    return quota


def pick_from_bucket(items, quota, max_per_day):
    """בוחר את הטובות ביותר, עם תקרה ליום כדי שאירוע אחד לא ישתלט על השנה."""
    items = sorted(items, key=lambda r: r["score"], reverse=True)
    per_day = defaultdict(int)
    chosen, overflow = [], []
    for r in items:
        if len(chosen) >= quota:
            break
        day = r["datetime"][:10]
        if per_day[day] >= max_per_day:
            overflow.append(r)
            continue
        chosen.append(r)
        per_day[day] += 1
    # אם התקרה היומית מנעה מילוי המכסה, ממלאים מהעודפים
    for r in overflow:
        if len(chosen) >= quota:
            break
        chosen.append(r)
    return chosen


# ---------------------------------------------------------------- ראשי

def main():
    ap = argparse.ArgumentParser(
        description="בורר תמונות מגוונות לסרטון בת מצווה",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="תיקיית התמונות (נסרקת רקורסיבית)")
    ap.add_argument("--output", default="./selected", help="תיקיית הפלט")
    ap.add_argument("--birthdate", required=True, help="תאריך לידה, YYYY-MM-DD")
    ap.add_argument("--total", type=int, default=300, help="כמה תמונות לבחור")
    ap.add_argument("--min-per-year", type=int, default=12,
                    help="מינימום תמונות מכל שנת גיל")
    ap.add_argument("--max-per-day", type=int, default=4,
                    help="מקסימום תמונות מאותו יום")
    ap.add_argument("--min-sharpness", type=float, default=45.0,
                    help="סף חדות מוחלט — מתחתיו התמונה נפסלת כמטושטשת")
    ap.add_argument("--no-faces", action="store_true",
                    help="לדלג על זיהוי פנים (מהיר יותר, פחות מדויק)")
    ap.add_argument("--hash-threshold", type=int, default=8,
                    help="רגישות זיהוי כפילויות. גבוה יותר = אגרסיבי יותר")
    ap.add_argument("--move", action="store_true",
                    help="להעביר קבצים במקום להעתיק")
    ap.add_argument("--dry-run", action="store_true",
                    help="רק דוח CSV, בלי להעתיק קבצים")
    ap.add_argument("--rescan", action="store_true", help="להתעלם מה-cache ולנתח מחדש")
    ap.add_argument("--w-sharp", type=float, default=0.30)
    ap.add_argument("--w-face", type=float, default=0.40)
    ap.add_argument("--w-expo", type=float, default=0.20)
    ap.add_argument("--w-res", type=float, default=0.10)
    args = ap.parse_args()

    root = Path(args.input).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"התיקייה לא נמצאה: {root}")

    try:
        birthdate = datetime.strptime(args.birthdate, "%Y-%m-%d")
    except ValueError:
        sys.exit("פורמט תאריך לידה שגוי. דוגמה: --birthdate 2014-03-21")

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / CACHE_NAME

    # ---- איסוף קבצים
    all_files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in all_files if p.suffix.lower() in IMAGE_EXT]
    videos = [p for p in all_files if p.suffix.lower() in VIDEO_EXT]

    heic_count = sum(1 for p in images if p.suffix.lower() in {".heic", ".heif"})
    if heic_count and not HEIC_OK:
        print(f"אזהרה: {heic_count} קובצי HEIC יידלגו. התקן: pip install pillow-heif")
        images = [p for p in images if p.suffix.lower() not in {".heic", ".heif"}]

    print(f"נמצאו {len(images)} תמונות ו-{len(videos)} סרטונים (סרטונים לא מטופלים).")
    if not images:
        sys.exit("אין תמונות לעבד.")

    # ---- ניתוח (עם cache)
    records = []
    cache = {}
    if cache_path.exists() and not args.rescan:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = {r["path"]: r for r in json.load(f)}
            print(f"נטענו {len(cache)} תוצאות ניתוח מה-cache.")
        except Exception:
            cache = {}

    todo = [p for p in images if str(p) not in cache]
    if todo:
        takeout_index = build_takeout_index(root)
        if takeout_index:
            print(f"נמצאו תאריכי צילום מקוריים ל-{len(takeout_index)} קבצים.")
        else:
            print("לא נמצא metadata של Takeout — נופל ל-EXIF ולשמות קבצים.")
        for p in tqdm(todo, desc="מנתח תמונות"):
            rec = analyze_image(p, takeout_index, detect_faces=not args.no_faces)
            if rec:
                cache[str(p)] = rec
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(list(cache.values()), f, ensure_ascii=False)

    records = [cache[str(p)] for p in images if str(p) in cache]
    print(f"נותחו בהצלחה {len(records)} תמונות.")

    # ---- ניקוד
    records = compute_scores(records, not args.no_faces,
                             args.w_sharp, args.w_face, args.w_expo, args.w_res)

    # ---- סינון חדות מוחלט
    before = len(records)
    records = [r for r in records if r["sharpness"] >= args.min_sharpness]
    print(f"נפסלו {before - len(records)} תמונות מטושטשות (סף {args.min_sharpness}).")

    # ---- כפילויות
    kept, dropped = dedupe(records, hash_threshold=args.hash_threshold)
    print(f"הוסרו {len(dropped)} כפילויות/בּרסטים. נותרו {len(kept)} מועמדות.")

    # ---- דליים לפי גיל
    buckets = defaultdict(list)
    out_of_range = 0
    for r in kept:
        dt = datetime.fromisoformat(r["datetime"])
        age = age_at(dt, birthdate)
        if age < 0 or age > 13:
            out_of_range += 1
            continue
        r["age"] = age
        buckets[age].append(r)
    if out_of_range:
        print(f"שים לב: {out_of_range} תמונות מחוץ לטווח 0-13 — כנראה תאריך שגוי.")

    print("\nזמינות לפי גיל:")
    for age in sorted(buckets):
        print(f"  גיל {age:>2}: {len(buckets[age]):>4} מועמדות")

    quotas = allocate_quotas(buckets, args.total, args.min_per_year)

    selected = []
    for age in sorted(buckets):
        picks = pick_from_bucket(buckets[age], quotas[age], args.max_per_day)
        selected.extend(picks)

    selected.sort(key=lambda r: r["datetime"])
    print(f"\nנבחרו {len(selected)} תמונות מתוך {args.total} מבוקשות.")
    print("\nהתפלגות סופית:")
    dist = defaultdict(int)
    for r in selected:
        dist[r["age"]] += 1
    for age in sorted(dist):
        bar = "█" * max(1, dist[age] // 2)
        print(f"  גיל {age:>2}: {dist[age]:>3}  {bar}")

    # ---- דוח CSV
    csv_path = out_dir / "selection_report.csv"
    fields = ["seq", "age", "datetime", "date_source", "score", "orientation",
              "width", "height", "face_count", "sharpness", "cluster_size",
              "name", "path"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(selected, 1):
            row = dict(r)
            row["seq"] = i
            row.setdefault("cluster_size", 1)
            w.writerow(row)
    print(f"\nדוח נשמר: {csv_path}")

    if args.dry_run:
        print("dry-run — לא הועתקו קבצים.")
        return

    # ---- העתקה, ממוינת לפי גיל וסדר כרונולוגי
    for i, r in enumerate(selected, 1):
        src = Path(r["path"])
        age_dir = out_dir / f"age_{r['age']:02d}"
        age_dir.mkdir(exist_ok=True)
        dst = age_dir / f"{i:03d}_{r['datetime'][:10]}_s{int(r['score']):02d}_{src.name}"
        try:
            if args.move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
        except Exception as e:
            print(f"שגיאה בהעתקת {src.name}: {e}")

    print(f"הקבצים הועתקו אל: {out_dir}")
    print("\nהמלצה: עבור על התיקיות לפי סדר גיל ומחק מה שלא מתאים — "
          "ואז הרץ שוב עם --total גבוה יותר אם חסרות תמונות.")


if __name__ == "__main__":
    main()
