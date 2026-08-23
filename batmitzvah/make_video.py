#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_video.py — מרכיב סרטון בת מצווה מתמונות + פסקול
=====================================================

מקבל תיקיית תמונות (למשל הפלט של select_photos.py) ועד שלושה קובצי שיר,
ומחזיר קובץ MP4 אחד: כל תמונה 2 שניות, מעברים רכים, פסקול עם קרוס-פייד
בין השירים ופייד-אאוט בסוף.

התקנה:
    pip install pillow pillow-heif python-bidi imageio-ffmpeg

הרצה בסיסית:
    python3 make_video.py --photos ./photos --out batmitzvah.mp4 \
        --song "love_story.m4a" --song "yesh_bach_hakol.mp3" --song "rak_ma_sheat_ohevet.mp3"

הרצת בדיקה מהירה (10 תמונות ראשונות):
    python3 make_video.py --photos ./photos --out test.mp4 --preview 10

חשוב: קובצי השירים הם באחריותך — הסקריפט לא מוריד מוזיקה מהאינטרנט.
ייצא/הקלט אותם מהמקור החוקי שלך (Apple Music/Spotify לא מאפשרים ייצוא;
קנייה ב-iTunes או קובץ שיש לך כבר עובדים מצוין).
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
except ImportError:
    sys.exit("חסרה Pillow. הרץ: pip install pillow pillow-heif")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # רק אומר שקובצי HEIC לא ייקראו

try:
    from bidi.algorithm import get_display
except ImportError:
    def get_display(s):          # כתוביות בעברית ייצאו הפוכות בלי python-bidi
        return s

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma", ".aif", ".aiff"}

# מעברים מתחלפים במחזוריות קבועה — נעים לעין ולא מקפיץ
MIXED_TRANSITIONS = ["fade", "fade", "smoothleft", "fade", "fade", "smoothright",
                     "fade", "dissolve", "fade", "slideleft"]


# ------------------------------------------------------------------ עזרי ffmpeg

def ffmpeg_bin():
    """מאתר ffmpeg: משתנה סביבה, PATH, ואז הבינארי המצורף ל-imageio-ffmpeg."""
    env = os.environ.get("FFMPEG")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("לא נמצא ffmpeg. הרץ: pip install imageio-ffmpeg  (או התקן ffmpeg במערכת)")


FFMPEG = None


def run(cmd, what=""):
    """מריץ ffmpeg ומרים שגיאה עם הזנב של stderr — שם נמצא המידע המעניין."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-15:]
        raise RuntimeError(f"ffmpeg נכשל{' ב' + what if what else ''}:\n" + "\n".join(tail))
    return proc


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")


def media_duration(path):
    """אורך קובץ מדיה בשניות, או None אם לא זוהה."""
    proc = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    m = _DUR_RE.search(proc.stderr.decode("utf-8", "replace"))
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


# ------------------------------------------------------------------ איסוף תמונות

def natural_key(name):
    """מיון טבעי: 9 לפני 10, ולא לקסיקוגרפי."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def collect_photos(photos_dir, order_csv, limit):
    photos_dir = Path(photos_dir).expanduser().resolve()
    if not photos_dir.is_dir():
        sys.exit(f"תיקיית התמונות לא נמצאה: {photos_dir}")

    if order_csv:
        # מכבד את הסדר הכרונולוגי מהדוח של select_photos.py
        csv_path = Path(order_csv).expanduser().resolve()
        if not csv_path.is_file():
            sys.exit(f"קובץ הסדר לא נמצא: {csv_path}")
        by_name = {}
        for p in photos_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXT:
                by_name.setdefault(p.name, p)
                # שמות הפלט של select_photos.py: 001_2014-04-02_s71_IMG_0031.jpg
                m = re.match(r"^\d{3,}_\d{4}-\d{2}-\d{2}_s\d+_(.+)$", p.name)
                if m:
                    by_name.setdefault(m.group(1), p)
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r.get("seq") or 0))
        files, missing = [], 0
        for r in rows:
            hit = by_name.get(Path(r.get("name", "")).name)
            if hit:
                files.append(hit)
            else:
                missing += 1
        if missing:
            print(f"אזהרה: {missing} שורות מה-CSV לא נמצאו בתיקייה — דולגו.")
    else:
        files = sorted((p for p in photos_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXT),
                       key=lambda p: (natural_key(str(p.parent)), natural_key(p.name)))

    if not files:
        sys.exit(f"לא נמצאו תמונות ב-{photos_dir}")
    if limit:
        files = files[:limit]
    return files


# ------------------------------------------------------------------ נרמול תמונה

def normalize_photo(src, dst, out_w, out_h, oversample, background):
    """
    מיישר לפי EXIF, ממיר ל-RGB, וממקם על קנבס ביחס הגובה-רוחב של הפלט.
    מחזיר True בהצלחה. תמונות פורטרט מקבלות רקע מטושטש מעצמן ולא פסים שחורים.
    """
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception as e:
        print(f"  דילוג על {Path(src).name}: {e}")
        return False

    cw, ch = int(out_w * oversample), int(out_h * oversample)
    cw += cw % 2
    ch += ch % 2

    if background == "blur":
        canvas = ImageOps.fit(img, (cw, ch), method=Image.LANCZOS, centering=(0.5, 0.4))
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(cw, ch) // 45))
        canvas = Image.blend(canvas, Image.new("RGB", (cw, ch), (12, 12, 14)), 0.28)
    else:
        canvas = Image.new("RGB", (cw, ch), (0, 0, 0) if background == "black" else (255, 255, 255))

    fitted = img.copy()
    fitted.thumbnail((cw, ch), Image.LANCZOS)
    canvas.paste(fitted, ((cw - fitted.width) // 2, (ch - fitted.height) // 2))
    canvas.save(dst, "JPEG", quality=95, subsampling=0)
    return True


def make_title_card(text, subtext, dst, out_w, out_h, oversample, font_path):
    """כרטיס כותרת בעברית. הטקסט עובר bidi כדי שלא ייצא הפוך."""
    cw, ch = int(out_w * oversample), int(out_h * oversample)
    img = Image.new("RGB", (cw, ch), (10, 10, 14))
    draw = ImageDraw.Draw(img)

    def load(size):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            return ImageFont.load_default()

    y = ch // 2
    if text:
        f = load(int(ch * 0.11))
        t = get_display(text)
        w = draw.textbbox((0, 0), t, font=f)[2]
        draw.text(((cw - w) // 2, y - int(ch * 0.10)), t, font=f, fill=(250, 246, 238))
    if subtext:
        f = load(int(ch * 0.045))
        t = get_display(subtext)
        w = draw.textbbox((0, 0), t, font=f)[2]
        draw.text(((cw - w) // 2, y + int(ch * 0.04)), t, font=f, fill=(198, 176, 140))
    img.save(dst, "JPEG", quality=95, subsampling=0)


# ------------------------------------------------------------------ קליפ בודד

def render_clip(idx, img_path, duration, out_path, out_w, out_h, fps, zoom):
    """
    הופך תמונה בודדת לקליפ וידאו. zoompan נותן אפקט קן-ברנס עדין —
    כל קליפ מקבל כיוון אחר לפי המקום שלו ברצף, כך שאין תחושת לופ.
    """
    frames = max(1, int(round(duration * fps)))
    if zoom:
        mode = idx % 4
        rate = 0.0009
        if mode in (0, 2):   # זום פנימה
            z = f"min(1+{rate}*on,1.14)"
        else:                # זום החוצה
            z = f"max(1.14-{rate}*on,1.0)"
        if mode < 2:         # מרכז
            x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        else:                # הסחה קלה, נותן תנועה אלכסונית
            x, y = "iw/2-(iw/zoom/2)+(on/%d)*40" % frames, "ih/2-(ih/zoom/2)-(on/%d)*25" % frames
        vf = (f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={out_w}x{out_h}:fps={fps},"
              f"format=yuv420p")
    else:
        vf = f"scale={out_w}:{out_h}:flags=lanczos,format=yuv420p"

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-loop", "1", "-framerate", str(fps), "-i", str(img_path),
           "-t", f"{duration:.4f}", "-vf", vf, "-r", str(fps),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
           "-pix_fmt", "yuv420p", "-an", str(out_path)]
    run(cmd, f"רינדור {Path(img_path).name}")
    return out_path


# ------------------------------------------------------------------ שרשור וידאו

def concat_hard_cuts(clips, out_path):
    """חיתוכים ישרים — מהיר, בלי קידוד מחדש."""
    listfile = out_path.parent / "concat_list.txt"
    with open(listfile, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{str(c)}'\n")
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
         "-safe", "0", "-i", str(listfile), "-c", "copy", str(out_path)], "שרשור")
    return out_path


def concat_with_xfade(clips, budgets, xfade, style, fps, out_path, batch=25):
    """
    שרשור עם מעברים רכים. שרשרת xfade של 100 קליפים בפקודה אחת עובדת אבל
    כבדה מאוד, ולכן מפצלים לקבוצות ואז משרשרים את הקבוצות — גם כן ב-xfade,
    כך שאין חיתוך יבש בין קבוצות.
    """
    if len(clips) == 1:
        shutil.copy2(clips[0], out_path)
        return out_path

    def xfade_chain(items, item_budgets, dst, lead=0.0):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
        for c in items:
            cmd += ["-i", str(c)]
        parts, prev, offset = [], "0:v", lead
        for i in range(1, len(items)):
            offset += item_budgets[i - 1]
            trans = MIXED_TRANSITIONS[(i - 1) % len(MIXED_TRANSITIONS)] if style == "mixed" else style
            label = f"v{i}"
            parts.append(f"[{prev}][{i}:v]xfade=transition={trans}:"
                         f"duration={xfade:.4f}:offset={offset - xfade:.4f}[{label}]")
            prev = label
        cmd += ["-filter_complex", ";".join(parts), "-map", f"[{prev}]",
                "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an", str(dst)]
        run(cmd, "מעברים")
        return dst

    if len(clips) <= batch:
        return xfade_chain(clips, budgets, out_path)

    # שלב א: קבוצות. הקליפ הראשון בכל קבוצה (פרט לראשונה) ארוך ב-xfade
    # אחד בהתחלה — החפיפה עם הקבוצה שלפניה — ולכן אורך הקבוצה בפועל הוא
    # סכום התקציבים ועוד xfade, בדיוק מה ששלב ב מצפה לו.
    tmpdir = out_path.parent
    groups, gbudgets = [], []
    i = 0
    while i < len(clips):
        chunk = clips[i:i + batch]
        chunk_b = budgets[i:i + batch]
        dst = tmpdir / f"group_{len(groups):03d}.mp4"
        if len(chunk) == 1:
            shutil.copy2(chunk[0], dst)
        else:
            # קבוצה שאינה הראשונה מתחילה בקליפ שארוך ב-xfade אחד בהתחלה
            xfade_chain(chunk, chunk_b, dst, lead=0.0 if i == 0 else xfade)
        groups.append(dst)
        gbudgets.append(sum(chunk_b))
        i += batch
        print(f"  קבוצה {len(groups)}/{-(-len(clips)//batch)} הושלמה")
    return xfade_chain(groups, gbudgets, out_path)


# ------------------------------------------------------------------ פסקול

def parse_song(spec):
    """
    'file.mp3'            — כל השיר, מההתחלה
    'file.mp3:45'         — מתחיל בשנייה 45
    'file.mp3:45:70'      — מתחיל ב-45, לוקח 70 שניות
    """
    path, start, dur = spec, 0.0, None
    m = re.match(r"^(.*?):(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?$", spec)
    if m and Path(m.group(1)).suffix.lower() in AUDIO_EXT:
        path = m.group(1)
        start = float(m.group(2))
        dur = float(m.group(3)) if m.group(3) else None
    p = Path(path).expanduser()
    if not p.is_file():
        sys.exit(f"קובץ השיר לא נמצא: {p}")
    return p, start, dur


def build_audio(songs, total_len, afade, fadeout, out_path):
    """
    חותך מכל שיר את הקטע המבוקש, משרשר עם קרוס-פייד, משווה עוצמות,
    וגומר בדיוק באורך הווידאו עם פייד-אאוט.
    """
    n = len(songs)
    # מי שלא קיבל אורך מפורש מתחלק שווה בשארית. חפיפות הקרוס-פייד
    # מתקצרות מהסכום, ולכן מוסיפים אותן חזרה לתקציב.
    fixed = sum(d for _, _, d in songs if d)
    free = [i for i, (_, _, d) in enumerate(songs) if not d]
    target_sum = total_len + (n - 1) * afade
    plan = []
    for i, (p, start, dur) in enumerate(songs):
        if dur is None:
            dur = max(4.0, (target_sum - fixed) / max(len(free), 1))
        avail = media_duration(p)
        if avail is not None:
            usable = max(1.0, avail - start)
            if dur > usable + 0.05:
                print(f"  אזהרה: {p.name} נותן רק {usable:.0f}ש' מ-{start:.0f} "
                      f"(ביקשנו {dur:.0f}) — מתקצר.")
                dur = usable
        plan.append((p, start, dur))

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p, start, dur in plan:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(p)]

    parts = []
    for i, (p, start, dur) in enumerate(plan):
        parts.append(f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                     f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:st=0:d=1.2[a{i}]")
    prev = "a0"
    for i in range(1, len(plan)):
        label = f"m{i}"
        parts.append(f"[{prev}][a{i}]acrossfade=d={afade:.3f}:c1=tri:c2=tri[{label}]")
        prev = label
    fade_start = max(0.0, total_len - fadeout)
    parts.append(f"[{prev}]apad,atrim=0:{total_len:.4f},"
                 f"afade=t=out:st={fade_start:.4f}:d={fadeout:.3f}[aout]")

    cmd += ["-filter_complex", ";".join(parts), "-map", "[aout]",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out_path)]
    run(cmd, "פסקול")

    total_used = sum(d for _, _, d in plan) - (n - 1) * afade
    print(f"  פסקול: " + " + ".join(f"{p.name} ({s:.0f}→{s+d:.0f})" for p, s, d in plan)
          + f" = {total_used:.0f}ש'")
    if total_used < total_len - 1:
        print(f"  אזהרה: המוזיקה קצרה ב-{total_len - total_used:.0f}ש' מהווידאו — "
              f"הסוף יהיה שקט. הוסף שיר או הארך קטעים.")
    return out_path


# ------------------------------------------------------------------ ראשי

def main():
    global FFMPEG
    ap = argparse.ArgumentParser(
        description="מרכיב סרטון בת מצווה מתמונות ופסקול",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--photos", required=True, help="תיקיית התמונות (נסרקת רקורסיבית)")
    ap.add_argument("--out", default="batmitzvah.mp4", help="קובץ הפלט")
    ap.add_argument("--song", action="append", default=[],
                    help="קובץ שיר. אפשר לחזור על הדגל. תחביר: file.mp3[:התחלה[:אורך]]")
    ap.add_argument("--per-photo", type=float, default=2.0, help="שניות לכל תמונה")
    ap.add_argument("--order-csv", help="selection_report.csv — לשמירת הסדר הכרונולוגי")
    ap.add_argument("--preview", type=int, default=0, help="להשתמש רק ב-N התמונות הראשונות")
    ap.add_argument("--size", default="1920x1080", help="רזולוציית פלט, WxH")
    ap.add_argument("--fps", type=int, default=30, help="פריימים בשנייה")
    ap.add_argument("--transition", default="mixed",
                    help="mixed / fade / dissolve / smoothleft / none")
    ap.add_argument("--xfade", type=float, default=0.6, help="אורך המעבר בשניות")
    ap.add_argument("--no-zoom", action="store_true", help="בלי אפקט קן-ברנס")
    ap.add_argument("--background", default="blur", choices=["blur", "black", "white"],
                    help="מה למלא בצדדים של תמונות פורטרט")
    ap.add_argument("--title", help="כרטיס פתיחה — טקסט ראשי")
    ap.add_argument("--subtitle", help="כרטיס פתיחה — טקסט משני")
    ap.add_argument("--end-title", help="כרטיס סיום")
    ap.add_argument("--title-duration", type=float, default=3.5, help="אורך כרטיס כותרת")
    ap.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    help="גופן לכותרות (צריך לתמוך בעברית)")
    ap.add_argument("--audio-xfade", type=float, default=2.5, help="קרוס-פייד בין שירים")
    ap.add_argument("--fadeout", type=float, default=4.0, help="פייד-אאוט בסוף הפסקול")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)),
                    help="כמה קליפים לרנדר במקביל")
    ap.add_argument("--keep-temp", action="store_true", help="לא למחוק קבצי עבודה")
    args = ap.parse_args()

    FFMPEG = ffmpeg_bin()
    try:
        out_w, out_h = (int(x) for x in args.size.lower().split("x"))
    except ValueError:
        sys.exit("פורמט רזולוציה שגוי. דוגמה: --size 1920x1080")
    out_w += out_w % 2
    out_h += out_h % 2

    photos = collect_photos(args.photos, args.order_csv, args.preview)
    songs = [parse_song(s) for s in args.song]
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zoom = not args.no_zoom
    oversample = 1.35 if zoom else 1.0
    xfade = 0.0 if args.transition == "none" else args.xfade
    if xfade >= args.per_photo:
        sys.exit(f"--xfade ({xfade}) חייב להיות קטן מ---per-photo ({args.per_photo})")

    print(f"ffmpeg: {FFMPEG}")
    print(f"{len(photos)} תמונות × {args.per_photo}ש' → "
          f"{len(photos) * args.per_photo:.0f}ש' של תמונות")

    tmpdir = Path(tempfile.mkdtemp(prefix="batmitzvah_", dir=out_path.parent))
    try:
        # ---- שלב 1: נרמול (EXIF, HEIC, יחס גובה-רוחב)
        norm_dir = tmpdir / "norm"
        norm_dir.mkdir()
        slides = []   # (נתיב_מנורמל, תקציב_שניות)

        if args.title or args.subtitle:
            card = norm_dir / "000_title.jpg"
            make_title_card(args.title, args.subtitle, card, out_w, out_h, oversample, args.font)
            slides.append((card, args.title_duration))

        print("מנרמל תמונות...")
        norm_paths = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = []
            for i, src in enumerate(photos, 1):
                dst = norm_dir / f"{i:04d}.jpg"
                futs.append((dst, pool.submit(normalize_photo, src, dst, out_w, out_h,
                                              oversample, args.background)))
            for dst, fut in futs:
                if fut.result():
                    norm_paths.append(dst)
        print(f"  {len(norm_paths)}/{len(photos)} תמונות עברו נרמול.")
        if not norm_paths:
            sys.exit("אף תמונה לא נקראה בהצלחה.")
        slides += [(p, args.per_photo) for p in norm_paths]

        if args.end_title:
            card = norm_dir / "zzz_end.jpg"
            make_title_card(args.end_title, None, card, out_w, out_h, oversample, args.font)
            slides.append((card, args.title_duration))

        total_len = sum(b for _, b in slides)

        # ---- שלב 2: קליפ לכל שקופית
        # הקליפ הראשון באורך התקציב שלו; כל השאר ארוכים ב-xfade אחד,
        # כדי שהחפיפות לא יקצרו את האורך הכולל. סה"כ = סכום התקציבים בדיוק.
        clip_dir = tmpdir / "clips"
        clip_dir.mkdir()
        print(f"מרנדר {len(slides)} קליפים ({args.jobs} במקביל)...")
        jobs = []
        for i, (img, budget) in enumerate(slides):
            dur = budget if i == 0 else budget + xfade
            jobs.append((i, img, dur, clip_dir / f"c{i:04d}.mp4"))
        clips = [None] * len(jobs)
        done = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(render_clip, i, img, dur, dst, out_w, out_h,
                                args.fps, zoom): i for i, img, dur, dst in jobs}
            for fut, i in futs.items():
                clips[i] = fut.result()
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}")

        # ---- שלב 3: שרשור
        budgets = [b for _, b in slides]
        silent = tmpdir / "silent.mp4"
        if xfade <= 0:
            print("משרשר בחיתוכים ישרים...")
            concat_hard_cuts(clips, silent)
        else:
            print(f"משרשר עם מעברי {args.transition} ({xfade}ש')...")
            concat_with_xfade(clips, budgets, xfade, args.transition, args.fps, silent)

        actual = media_duration(silent)
        print(f"  אורך הווידאו: {actual:.1f}ש' (יעד {total_len:.1f}ש')")
        video_len = actual or total_len

        # ---- שלב 4: פסקול ומיקס
        if songs:
            print("בונה פסקול...")
            audio = tmpdir / "audio.m4a"
            build_audio(songs, video_len, args.audio_xfade, args.fadeout, audio)
            print("ממזג...")
            run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(silent), "-i", str(audio),
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "192k", "-shortest",
                 "-movflags", "+faststart", str(out_path)], "מיזוג")
        else:
            print("בלי מוזיקה (לא הועבר --song).")
            run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent),
                 "-c", "copy", "-movflags", "+faststart", str(out_path)], "העברה")

        size_mb = out_path.stat().st_size / 1e6
        mins, secs = divmod(int(round(video_len)), 60)
        print(f"\nמוכן: {out_path}")
        print(f"  {mins}:{secs:02d} · {out_w}x{out_h} · {args.fps}fps · {size_mb:.0f}MB")
    finally:
        if args.keep_temp:
            print(f"קבצי עבודה נשארו ב: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
