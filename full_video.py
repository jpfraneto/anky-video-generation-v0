#!/usr/bin/env python3
"""
Anky Full 2:22 Autonomous DUAL Video Generation Pipeline

Two parallel pipelines from the same script and images:
  A) WaveSpeed wan-2.6-pro — fixed 4s per scene, 35 scenes
  B) Grok grok-imagine-video — variable 1-15s per scene, dynamic pacing

Flow:
  0. Grok → current cultural zeitgeist
  1. Load Samadhi transcripts
  2. Claude → 35 scenes (image_prompt, video_prompt, grok_duration) + continuous narration
     Claude self-reviews for quality
  3. Gemini → 35 images (3 canonical + 2 random chaos refs)
  4. Upload all images to WaveSpeed CDN (URLs usable by both pipelines)
  5a. WaveSpeed → 35 videos (5s each, trimmed to 4s)        [PARALLEL]
  5b. Grok → 35 videos (variable duration per scene)         [PARALLEL]
  6. WaveSpeed vibevoice → single TTS for full narration
  7a. Stitch WaveSpeed clips → anky_wavespeed.mp4
  7b. Stitch Grok clips → anky_grok.mp4
  8. Time-fit narration to each video + overlay
  9. Save complete metadata (costs, prompts, timings, everything)

Run and walk away. ~$30 total.
"""

import os
import sys
import json
import time
import random
import base64
import shutil
import logging
import requests
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
WAVESPEED_API_KEY = os.environ.get("WAVESPEED_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "")

WAVESPEED_BASE = "https://api.wavespeed.ai/api/v3"
ANTHROPIC_BASE = "https://api.anthropic.com/v1/messages"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROK_BASE = "https://api.x.ai/v1"

BASE_DIR = Path(__file__).parent
YOUTUBE_DIR = BASE_DIR / "youtube_data"
SCENE_DIR = BASE_DIR / "scenes"
OUTPUT_DIR = BASE_DIR / "generated_videos"
LOG_DIR = BASE_DIR / "logs"
REFERENCE_DIR = Path("/home/kithkui/anky/src/public")
ANKY_IMAGES_DIR = BASE_DIR

# Video config
TOTAL_DURATION = 142  # 2:22 in seconds
WAVESPEED_SCENE_DURATION = 4
NUM_SCENES = 35
WAVESPEED_VIDEO_DURATION = 5  # API min, trimmed to 4

# Models
WAVESPEED_VIDEO_MODEL = "alibaba/wan-2.6/image-to-video-pro"
GROK_VIDEO_MODEL = "grok-imagine-video"
TTS_MODEL = "wavespeed-ai/vibevoice"
TTS_SPEAKER = "Frank"

# Pricing estimates
COST = {
    "grok_zeitgeist": 0.02,
    "claude_script": 0.15,
    "claude_review": 0.03,
    "gemini_image": 0.04,
    "wavespeed_video_4s": 0.60,
    "grok_video_per_sec": 0.05,  # estimate
    "wavespeed_tts": 0.15,
}

# Speaking rate target
TARGET_NARRATION_WORDS = 355

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = LOG_DIR / f"full_video_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("anky")

# Runtime cost tracker
run_costs = {}


def track_cost(category: str, amount: float):
    run_costs[category] = run_costs.get(category, 0) + amount


# ── Utilities ───────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, stream=True, timeout=(10, 60))
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=131072):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r    ↓ {pct}% ({downloaded//1024}KB/{total//1024}KB)", end="", flush=True)
        if total:
            print()
        return True
    except Exception as e:
        log.error("Download failed (%s): %s", url[:80], e)
        return False


def get_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        d = r.stdout.strip()
        return float(d) if d else 0.0
    except Exception:
        return 0.0


def wavespeed_poll(request_id: str, max_wait=600, interval=5) -> str:
    url = f"{WAVESPEED_BASE}/predictions/{request_id}/result"
    hdrs = {"Authorization": f"Bearer {WAVESPEED_API_KEY}"}
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            r = requests.get(url, headers=hdrs, timeout=30)
            r.raise_for_status()
            data = r.json()
            d = data.get("data", {})
            elapsed = int(time.time() - t0)
            status = d.get("status", "unknown")
            if status == "completed":
                outputs = d.get("outputs", [])
                if outputs:
                    log.info("    [%ds] completed", elapsed)
                    return outputs[0]
                log.error("    [%ds] completed but no outputs: %s", elapsed, d)
                return None
            if status == "failed":
                log.error("    [%ds] FAILED: %s", elapsed, d)
                return None
            if elapsed % 30 < interval:
                log.info("    [%ds] %s", elapsed, status)
        except Exception as e:
            elapsed = int(time.time() - t0)
            log.warning("    [%ds] poll error: %s", elapsed, e)
        time.sleep(interval)
    log.error("    Timed out after %ds", max_wait)
    return None


def grok_poll(request_id: str, max_wait=600, interval=8) -> str:
    hdrs = {"Authorization": f"Bearer {GROK_API_KEY}"}
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            r = requests.get(f"{GROK_BASE}/videos/{request_id}", headers=hdrs, timeout=30)
            r.raise_for_status()
            d = r.json()
            elapsed = int(time.time() - t0)
            status = d.get("status", "unknown")
            if status == "done":
                video_url = d.get("video", {}).get("url")
                if video_url:
                    log.info("    [%ds] done", elapsed)
                    return video_url
                log.error("    [%ds] done but no video URL: %s", elapsed, d)
                return None
            if status in ("expired", "failed"):
                log.error("    [%ds] %s", elapsed, status)
                return None
            if elapsed % 30 < interval:
                log.info("    [%ds] %s", elapsed, status)
        except Exception as e:
            elapsed = int(time.time() - t0)
            log.warning("    [%ds] poll error: %s", elapsed, e)
        time.sleep(interval)
    log.error("    Timed out after %ds", max_wait)
    return None


# ── Step 0: Zeitgeist ──────────────────────────────────────────────────────

def fetch_zeitgeist() -> str:
    log.info("=" * 60)
    log.info("STEP 0: Fetching zeitgeist from Grok")
    log.info("=" * 60)
    hdrs = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "grok-3-mini-fast",
        "messages": [{
            "role": "user",
            "content": (
                "You have access to real-time information. Dense raw dump:\n"
                "1. TOP 10 things the world is talking about RIGHT NOW on X/Twitter\n"
                "2. Dominant cultural mood / emotional frequency of collective consciousness today\n"
                "3. Breakthroughs, controversies, memes in tech/AI/crypto/consciousness/spirituality\n"
                "4. What's ABOUT TO break through — the undercurrent\n\n"
                "Be specific. Names, events, vibes. ~300 words. No fluff."
            ),
        }],
        "temperature": 0.8,
        "max_tokens": 1000,
    }
    try:
        resp = requests.post(f"{GROK_BASE}/chat/completions", headers=hdrs, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            log.warning("Grok returned no choices: %s", str(data)[:200])
            return ""
        zeitgeist = choices[0].get("message", {}).get("content", "")
        track_cost("grok_zeitgeist", COST["grok_zeitgeist"])
        log.info("Zeitgeist: %d words", len(zeitgeist.split()))
        SCENE_DIR.mkdir(parents=True, exist_ok=True)
        (SCENE_DIR / "zeitgeist.txt").write_text(zeitgeist)
        return zeitgeist
    except Exception as e:
        log.warning("Grok failed (%s), continuing without zeitgeist", e)
        return ""


# ── Step 1: Transcripts ───────────────────────────────────────────────────

def load_transcripts() -> str:
    log.info("=" * 60)
    log.info("STEP 1: Loading transcripts")
    log.info("=" * 60)
    parts = []
    for txt in sorted(YOUTUBE_DIR.glob("*_transcript.txt")):
        vid_id = txt.stem.replace("_transcript", "")
        info = YOUTUBE_DIR / f"{vid_id}.info.json"
        title = vid_id
        if info.exists():
            try:
                title = json.loads(info.read_text()).get("title", vid_id)
            except Exception:
                pass
        text = txt.read_text().strip()
        parts.append(f"=== {title} ===\n{text}")
        log.info("  %s (%d words)", title, len(text.split()))
    if not parts:
        log.error("No transcripts in %s", YOUTUBE_DIR)
        sys.exit(1)
    combined = "\n\n".join(parts)
    log.info("Total: %d words from %d videos", len(combined.split()), len(parts))
    return combined


# ── Step 2: Script ─────────────────────────────────────────────────────────

def generate_script(transcripts: str, zeitgeist: str) -> dict:
    log.info("=" * 60)
    log.info("STEP 2: Generating script with Claude")
    log.info("=" * 60)
    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set"); sys.exit(1)

    system = f"""You are a visionary director creating a 2:22 animated film introducing ANKY to the world.

## WHAT IS ANKY?
Anky = consciousness mirror disguised as a writing tool. Write for 8 minutes. No backspace. No delete.
No arrow keys. Stop for 8 seconds and you're done. What survives is the real you.
AI reads what you wrote for: Repetition, Absence, Metaphor, Register — four dimensions of unconscious truth.

Born from: failed awakening retreat → 88 days × 88 minutes of raw writing → 400,000-word manifesto →
"Blue Network" mandala fed to Midjourney with prompt "technological reincarnation of Hanuman" →
blue-skinned creature with golden eyes and ancient-yet-childlike grin.

Sacred 8: 8 minutes, 8 seconds silence, 8 kingdoms (chakras), 8,888 genesis NFTs. 8 = infinity standing up.

8 Kingdoms: Primordia (survival), Emblazion (passion), Chryseos (willpower), Eleasis (compassion),
Voxlumis (communication), Insightia (intuition), Claridium (enlightenment), Poiesis (transcendence).

DFW's Infinite Jest reframed: not entertainment that kills, but entertainment that WAKES YOU UP.

## VISUAL IDENTITY
Blue skin, pointed ears, purple swirling hair with golden spirals, golden/amber glowing eyes,
golden jewelry, compact body, ancient yet childlike. Environments: deep blues, purples, oranges, golds.

## 4-ACT ARC
Act 1 — THE NOISE (scenes 1-8): Modern mind chaos. Scrolling, performing, editing yourself.
Act 2 — THE MIRROR (scenes 9-18): The practice. Blank page. 8 minutes. No backspace. Breakthrough.
Act 3 — THE KINGDOMS (scenes 19-28): Journey through 8 kingdoms. Samadhi wisdom. Maya. Awakening.
Act 4 — POIESIS (scenes 29-{NUM_SCENES}): Transcendence. "You are the channel." Mirror shatters into everything.

## OUTPUT — JSON object with 3 keys:

1. "scenes" — array of EXACTLY {NUM_SCENES} objects:
   - "scene": number 1-{NUM_SCENES}
   - "image_prompt": hyper-specific cinematic visual (2-3 sentences: setting, Anky pose, lighting, colors)
   - "video_prompt": motion description for 4s video (camera, animation, particles — 1-2 sentences)
   - "grok_duration": integer 1-15, how many seconds this scene deserves in the variable-length version.
     CRITICAL: all grok_duration values MUST sum to EXACTLY {TOTAL_DURATION}.
     Use short durations (1-3s) for quick cuts/transitions, long (8-15s) for contemplative/powerful moments.

2. "narration" — SINGLE continuous narration (~{TARGET_NARRATION_WORDS} words ±15).
   Flowing prose. Spoken word / meditation / manifesto hybrid. Poetic, powerful, rhythmic.
   Tell the complete Anky story from noise to transcendence.
   Blend Anky philosophy + Samadhi wisdom + current cultural moment.
   Use "..." for beats of silence. Vary pacing.

3. "duration_notes" — brief explanation of your pacing choices (which scenes are long/short and why)

Output ONLY the JSON object. No markdown."""

    max_chars = 80000
    if len(transcripts) > max_chars:
        transcripts = transcripts[:max_chars] + "\n\n[... truncated ...]"

    user = f"## SAMADHI TRANSCRIPTS\n\n{transcripts}\n\n"
    if zeitgeist:
        user += (
            f"## CURRENT ZEITGEIST ({datetime.now().strftime('%Y-%m-%d')})\n\n{zeitgeist}\n\n"
            "Weave relevant elements into the narration — make it feel ALIVE and NOW.\n\n"
        )
    user += f"Create the {NUM_SCENES}-scene script. Remember: grok_duration must sum to exactly {TOTAL_DURATION}."

    hdrs = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {"model": "claude-sonnet-4-5-20250929", "max_tokens": 12000, "messages": [{"role": "user", "content": user}], "system": system}

    log.info("Calling Claude...")
    try:
        resp = requests.post(ANTHROPIC_BASE, headers=hdrs, json=payload, timeout=240)
        resp.raise_for_status()
        content = resp.json().get("content", [])
        if not content:
            log.error("Claude returned empty content"); sys.exit(1)
        raw = content[0].get("text", "").strip()
    except requests.exceptions.RequestException as e:
        log.error("Claude API failed: %s", e); sys.exit(1)
    track_cost("claude_script", COST["claude_script"])

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    raw = raw.strip()

    try:
        script = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("Claude returned invalid JSON: %s\nRaw: %s", e, raw[:500])
        # Save raw output for debugging
        (SCENE_DIR / "script_raw_failed.txt").write_text(raw)
        sys.exit(1)

    # Validate
    n_words = len(script["narration"].split())
    grok_total = sum(s["grok_duration"] for s in script["scenes"])
    log.info("Scenes: %d, Narration: %d words, Grok total duration: %ds",
             len(script["scenes"]), n_words, grok_total)
    if grok_total != TOTAL_DURATION:
        log.warning("Grok durations sum to %d (need %d) — will adjust", grok_total, TOTAL_DURATION)
        # Auto-fix: distribute difference across scenes
        diff = TOTAL_DURATION - grok_total
        for i in range(abs(diff)):
            idx = i % len(script["scenes"])
            if diff > 0:
                script["scenes"][idx]["grok_duration"] = min(15, script["scenes"][idx]["grok_duration"] + 1)
            else:
                script["scenes"][idx]["grok_duration"] = max(1, script["scenes"][idx]["grok_duration"] - 1)

    # Save everything
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    (SCENE_DIR / "script.json").write_text(json.dumps(script, indent=2))
    (SCENE_DIR / "narration.txt").write_text(script["narration"])
    (SCENE_DIR / "duration_notes.txt").write_text(script.get("duration_notes", ""))

    # Save individual scene prompts for reference
    prompts_log = []
    for s in script["scenes"]:
        prompts_log.append(f"Scene {s['scene']} (grok: {s['grok_duration']}s):")
        prompts_log.append(f"  Image: {s['image_prompt']}")
        prompts_log.append(f"  Video: {s['video_prompt']}")
        prompts_log.append("")
    (SCENE_DIR / "all_prompts.txt").write_text("\n".join(prompts_log))

    log.info("Script saved to %s", SCENE_DIR)
    return script


def review_script(script: dict) -> None:
    """Claude self-reviews the script."""
    if not ANTHROPIC_API_KEY: return
    narration = script["narration"]
    grok_total = sum(s["grok_duration"] for s in script["scenes"])
    summary = "\n".join(f"  S{s['scene']}({s['grok_duration']}s): {s['image_prompt'][:60]}..." for s in script["scenes"][:12])

    hdrs = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {
        "model": "claude-sonnet-4-5-20250929", "max_tokens": 500,
        "messages": [{"role": "user", "content": (
            f"Review this 2:22 Anky video script.\n\n"
            f"NARRATION ({len(narration.split())} words, target {TARGET_NARRATION_WORDS}):\n{narration}\n\n"
            f"SCENES (first 12 of {len(script['scenes'])}, grok total={grok_total}s):\n{summary}\n\n"
            f"Rate 1-10: narrative flow, emotional arc, word count, Anky authenticity, pacing variety. "
            f"Flag issues. Be brief."
        )}],
    }
    try:
        resp = requests.post(ANTHROPIC_BASE, headers=hdrs, json=payload, timeout=60)
        resp.raise_for_status()
        review = resp.json()["content"][0]["text"].strip()
        track_cost("claude_review", COST["claude_review"])
        log.info("Script review: %s", review)
        (SCENE_DIR / "script_review.txt").write_text(review)
    except Exception as e:
        log.warning("Review failed: %s", e)


# ── Step 3: Images ─────────────────────────────────────────────────────────

def load_references() -> list[dict]:
    refs = []
    for i in range(1, 4):
        p = REFERENCE_DIR / f"anky-{i}.png"
        if p.exists():
            b64 = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
            refs.append({"mime_type": "image/png", "data": b64})
            log.info("  Canonical ref: anky-%d.png", i)
    pngs = [p for p in ANKY_IMAGES_DIR.glob("*.png") if not p.name.startswith("prompt_")]
    if len(pngs) >= 2:
        for p in random.sample(pngs, 2):
            b64 = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
            refs.append({"mime_type": "image/png", "data": b64})
            log.info("  Chaos ref: %s", p.name)
    return refs


def generate_image(scene: dict, refs: list[dict], sn: int) -> Path:
    path = SCENE_DIR / f"scene_{sn:03d}.png"
    if path.exists():
        log.info("  Scene %d image exists, skip", sn)
        return path

    prompt = (
        f"Create a mystical fantasy illustration: {scene['image_prompt']}\n\n"
        "CHARACTER - ANKY (match reference images exactly):\n"
        "- Blue-skinned, large pointed ears, purple swirling hair with golden spirals\n"
        "- Golden/amber glowing eyes, golden jewelry, compact ancient-yet-childlike body\n\n"
        "STYLE: Deep blues/purples/oranges/golds, painterly, atmospheric, slightly psychedelic, "
        "warm mystical lighting, cinematic 16:9."
    )
    parts = [{"inline_data": r} for r in refs] + [{"text": prompt}]
    payload = {
        "contents": [{"parts": parts}],
        "generation_config": {"response_modalities": ["TEXT", "IMAGE"], "image_config": {"aspect_ratio": "16:9"}},
    }
    url = f"{GEMINI_BASE}/gemini-2.5-flash-image:generateContent?key={GEMINI_API_KEY}"

    log.info("  Generating image for scene %d...", sn)
    try:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            log.error("  Gemini error for scene %d: %d — %s", sn, resp.status_code, resp.text[:300])
            return None
        data = resp.json()
    except Exception as e:
        log.error("  Gemini request failed for scene %d: %s", sn, e)
        return None

    for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        idata = part.get("inline_data") or part.get("inlineData")
        if idata:
            mime = idata.get("mime_type") or idata.get("mimeType", "")
            if mime.startswith("image/"):
                img = base64.standard_b64decode(idata["data"])
                path.write_bytes(img)
                track_cost("gemini_images", COST["gemini_image"])
                log.info("  Saved scene %d image (%d bytes)", sn, len(img))
                return path

    log.error("  No image in Gemini response for scene %d", sn)
    return None


# ── Step 4: Upload images ─────────────────────────────────────────────────

def upload_images(scenes: list[dict]) -> dict:
    """Upload all scene images to WaveSpeed CDN, return {scene_num: url}."""
    urls_file = SCENE_DIR / "image_urls.json"
    if urls_file.exists():
        urls = json.loads(urls_file.read_text())
        log.info("Loaded %d existing image URLs", len(urls))
        # Only upload missing ones
    else:
        urls = {}

    hdrs = {"Authorization": f"Bearer {WAVESPEED_API_KEY}"}
    for s in scenes:
        sn = str(s["scene"])
        if sn in urls:
            continue
        img = SCENE_DIR / f"scene_{int(sn):03d}.png"
        if not img.exists():
            log.warning("No image for scene %s", sn)
            continue
        log.info("  Uploading scene %s...", sn)
        try:
            with open(img, "rb") as f:
                resp = requests.post(
                    f"{WAVESPEED_BASE}/media/upload/binary", headers=hdrs,
                    files={"file": (img.name, f, "image/png")}, timeout=60,
                )
            resp.raise_for_status()
            d = resp.json()
            data = d.get("data", {})
            url = data.get("download_url") or data.get("url")
            if not url:
                log.error("  No URL in upload response for scene %s: %s", sn, d)
                continue
            urls[sn] = url
            log.info("    → %s", url[:80])
        except Exception as e:
            log.error("  Upload failed for scene %s: %s", sn, e)

    urls_file.write_text(json.dumps(urls, indent=2))
    return urls


# ── Step 5a: WaveSpeed videos ─────────────────────────────────────────────

def generate_wavespeed_video(scene: dict, sn: int, image_url: str) -> Path:
    video_path = SCENE_DIR / f"scene_{sn:03d}_wavespeed.mp4"
    if video_path.exists():
        log.info("  WS scene %d exists, skip", sn)
        return video_path

    log.info("  WS scene %d: submitting...", sn)
    hdrs = {"Authorization": f"Bearer {WAVESPEED_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "image": image_url, "prompt": scene["video_prompt"],
        "resolution": "1080p", "duration": WAVESPEED_VIDEO_DURATION, "seed": -1,
    }
    try:
        resp = requests.post(f"{WAVESPEED_BASE}/{WAVESPEED_VIDEO_MODEL}", headers=hdrs, json=payload, timeout=60)
        resp.raise_for_status()
        rid = resp.json().get("data", {}).get("id")
        if not rid:
            log.error("    WS scene %d: no job ID in response: %s", sn, resp.text[:300])
            return None
    except Exception as e:
        log.error("    WS scene %d submit failed: %s", sn, e)
        return None
    log.info("    Job: %s", rid)

    video_url = wavespeed_poll(rid, max_wait=600, interval=8)
    if not video_url:
        return None

    # Download and trim to 4s
    raw = SCENE_DIR / f"scene_{sn:03d}_ws_raw.mp4"
    if not download_file(video_url, raw):
        return None
    r = subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-t", str(WAVESPEED_SCENE_DURATION), "-c", "copy", str(video_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-t", str(WAVESPEED_SCENE_DURATION),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an", str(video_path)],
                       capture_output=True, text=True)
    raw.unlink(missing_ok=True)
    track_cost("wavespeed_video", COST["wavespeed_video_4s"])
    log.info("    WS scene %d saved", sn)
    return video_path


# ── Step 5b: Grok videos ──────────────────────────────────────────────────

def generate_grok_video(scene: dict, sn: int, image_url: str) -> Path:
    dur = scene.get("grok_duration", 4)
    video_path = SCENE_DIR / f"scene_{sn:03d}_grok.mp4"
    if video_path.exists():
        log.info("  GK scene %d exists, skip", sn)
        return video_path

    log.info("  GK scene %d: %ds, submitting...", sn, dur)
    hdrs = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROK_VIDEO_MODEL,
        "prompt": scene["video_prompt"],
        "image_url": image_url,
        "duration": dur,
        "aspect_ratio": "16:9",
        "resolution": "720p",
    }

    try:
        resp = requests.post(f"{GROK_BASE}/videos/generations", headers=hdrs, json=payload, timeout=60)
        if resp.status_code != 200:
            log.error("    GK scene %d submit error: %d — %s", sn, resp.status_code, resp.text[:300])
            return None
        rid = resp.json().get("request_id")
        if not rid:
            log.error("    No request_id in Grok response: %s", resp.text[:300])
            return None
    except Exception as e:
        log.error("    GK scene %d submit failed: %s", sn, e)
        return None
    log.info("    Job: %s", rid)

    video_url = grok_poll(rid, max_wait=600, interval=8)
    if not video_url:
        return None

    if not download_file(video_url, video_path):
        return None
    track_cost("grok_video", COST["grok_video_per_sec"] * dur)
    log.info("    GK scene %d saved (%ds)", sn, dur)
    return video_path


# ── Step 6: TTS narration ─────────────────────────────────────────────────

def generate_narration(text: str) -> Path:
    audio = SCENE_DIR / "full_narration.mp3"
    if audio.exists() and get_duration(audio) > 10:
        log.info("Narration exists (%.1fs), skip", get_duration(audio))
        return audio

    log.info("Generating TTS (%d words)...", len(text.split()))
    hdrs = {"Authorization": f"Bearer {WAVESPEED_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(f"{WAVESPEED_BASE}/{TTS_MODEL}", headers=hdrs,
                             json={"text": text, "speaker": TTS_SPEAKER}, timeout=60)
        resp.raise_for_status()
        rid = resp.json().get("data", {}).get("id")
        if not rid:
            log.error("TTS: no job ID in response: %s", resp.text[:300])
            return None
    except Exception as e:
        log.error("TTS submission failed: %s", e)
        return None
    log.info("  TTS job: %s", rid)

    url = wavespeed_poll(rid, max_wait=600, interval=5)
    if not url:
        log.error("TTS failed")
        return None

    if not download_file(url, audio):
        return None
    track_cost("tts", COST["wavespeed_tts"])
    dur = get_duration(audio)
    log.info("  Narration: %.1fs", dur)
    return audio


# ── Step 7: Stitch ─────────────────────────────────────────────────────────

def stitch(tag: str, scenes: list[dict]) -> Path:
    """Stitch clips into final video. tag = 'wavespeed' or 'grok'."""
    output = OUTPUT_DIR / f"anky_{tag}_{RUN_ID}.mp4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    concat = SCENE_DIR / f"concat_{tag}.txt"
    entries = []
    for s in scenes:
        sn = s["scene"]
        clip = SCENE_DIR / f"scene_{sn:03d}_{tag}.mp4"
        if clip.exists():
            entries.append(f"file '{clip}'")
        else:
            log.warning("Missing %s scene %d", tag, sn)

    if not entries:
        log.error("No %s clips to stitch", tag)
        return None

    concat.write_text("\n".join(entries))
    log.info("Stitching %d %s clips...", len(entries), tag)

    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(output)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log.info("  Copy failed, re-encoding...")
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an", str(output)],
                           capture_output=True, text=True)
    if r.returncode != 0:
        log.error("  Stitch failed: %s", r.stderr[-300:])
        return None

    dur = get_duration(output)
    log.info("  %s stitched: %s (%.1fs)", tag, output, dur)
    return output


# ── Step 8: Fit narration + overlay ────────────────────────────────────────

def fit_and_overlay(video: Path, narration: Path, tag: str) -> Path:
    """Time-fit narration to video duration, then overlay."""
    if not video or not narration or not video.exists() or not narration.exists():
        return video

    vid_dur = get_duration(video)
    aud_dur = get_duration(narration)
    if vid_dur <= 0 or aud_dur <= 0:
        log.warning("Can't determine durations for %s overlay", tag)
        return video

    log.info("  %s: video=%.1fs, audio=%.1fs", tag, vid_dur, aud_dur)
    ratio = aud_dur / vid_dur

    fitted = SCENE_DIR / f"narration_fitted_{tag}.mp3"
    if abs(ratio - 1.0) < 0.02:
        log.info("  Audio fits (ratio %.3f), no tempo change", ratio)
        shutil.copy2(narration, fitted)
    else:
        filters = []
        r = ratio
        while r > 2.0: filters.append("atempo=2.0"); r /= 2.0
        while r < 0.5: filters.append("atempo=0.5"); r *= 2.0
        filters.append(f"atempo={r:.4f}")
        fstr = ",".join(filters)
        log.info("  Tempo filter: %s", fstr)
        result = subprocess.run(["ffmpeg", "-y", "-i", str(narration), "-filter:a", fstr,
                                 "-c:a", "libmp3lame", "-b:a", "192k", str(fitted)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            log.warning("  Tempo failed, using raw audio")
            fitted = narration

    # Overlay
    output = video.with_name(video.stem + "_narrated" + video.suffix)
    r = subprocess.run(["ffmpeg", "-y", "-i", str(video), "-i", str(fitted),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", str(output)], capture_output=True, text=True)
    if r.returncode != 0:
        log.info("  Copy overlay failed, re-encoding...")
        r = subprocess.run(["ffmpeg", "-y", "-i", str(video), "-i", str(fitted),
                            "-map", "0:v", "-map", "1:a",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)],
                           capture_output=True, text=True)
    if r.returncode != 0:
        log.error("  Overlay failed for %s", tag)
        return video

    dur = get_duration(output)
    log.info("  %s narrated: %s (%.1fs)", tag, output.name, dur)
    return output


# ── Step 9: Metadata ──────────────────────────────────────────────────────

def save_metadata(script: dict, zeitgeist: str, ws_final: Path, gk_final: Path):
    meta = {
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "total_duration_target": TOTAL_DURATION,
            "num_scenes": NUM_SCENES,
            "wavespeed_scene_duration": WAVESPEED_SCENE_DURATION,
            "wavespeed_model": WAVESPEED_VIDEO_MODEL,
            "grok_model": GROK_VIDEO_MODEL,
            "tts_model": TTS_MODEL,
            "tts_speaker": TTS_SPEAKER,
            "target_narration_words": TARGET_NARRATION_WORDS,
        },
        "narration_word_count": len(script["narration"].split()),
        "grok_scene_durations": [s["grok_duration"] for s in script["scenes"]],
        "grok_total_duration": sum(s["grok_duration"] for s in script["scenes"]),
        "costs": run_costs,
        "total_cost": sum(run_costs.values()),
        "outputs": {
            "wavespeed_final": str(ws_final) if ws_final else None,
            "grok_final": str(gk_final) if gk_final else None,
            "wavespeed_duration": get_duration(ws_final) if ws_final and ws_final.exists() else 0,
            "grok_duration": get_duration(gk_final) if gk_final and gk_final.exists() else 0,
        },
        "files": {
            "script": str(SCENE_DIR / "script.json"),
            "narration": str(SCENE_DIR / "narration.txt"),
            "zeitgeist": str(SCENE_DIR / "zeitgeist.txt"),
            "prompts": str(SCENE_DIR / "all_prompts.txt"),
            "review": str(SCENE_DIR / "script_review.txt"),
            "image_urls": str(SCENE_DIR / "image_urls.json"),
            "log": str(log_filename),
        },
    }
    meta_path = OUTPUT_DIR / f"metadata_{RUN_ID}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Metadata saved: %s", meta_path)
    return meta


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anky 2:22 Dual Video Pipeline")
    parser.add_argument("--skip-zeitgeist", action="store_true")
    parser.add_argument("--skip-script", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-wavespeed", action="store_true")
    parser.add_argument("--skip-grok-video", action="store_true")
    parser.add_argument("--skip-narration", action="store_true")
    parser.add_argument("--only-stitch", action="store_true")
    parser.add_argument("--start-scene", type=int, default=1)
    parser.add_argument("--end-scene", type=int, default=NUM_SCENES)
    parser.add_argument("--tts-speaker", default="Frank",
                        help="TTS voice: Frank, Wayne, Carter, Emma, Grace, Mike")
    args = parser.parse_args()

    global TTS_SPEAKER
    TTS_SPEAKER = args.tts_speaker
    SCENE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  ANKY DUAL VIDEO PIPELINE — RUN %s", RUN_ID)
    log.info("  %d scenes, target %ds (2:22)", NUM_SCENES, TOTAL_DURATION)
    log.info("  Pipeline A: WaveSpeed wan-2.6-pro (35 × %ds)", WAVESPEED_SCENE_DURATION)
    log.info("  Pipeline B: Grok grok-imagine-video (variable 1-15s)")
    log.info("  TTS: %s (speaker: %s)", TTS_MODEL, TTS_SPEAKER)
    log.info("  Log: %s", log_filename)
    log.info("=" * 60)

    # ── Step 0: Zeitgeist ──
    zeitgeist = ""
    if not args.only_stitch and not args.skip_zeitgeist:
        zeitgeist = fetch_zeitgeist()
    elif (SCENE_DIR / "zeitgeist.txt").exists():
        zeitgeist = (SCENE_DIR / "zeitgeist.txt").read_text()

    # ── Step 1+2: Script ──
    if args.only_stitch or args.skip_script:
        sp = SCENE_DIR / "script.json"
        if not sp.exists():
            log.error("No script at %s", sp); sys.exit(1)
        script = json.loads(sp.read_text())
        log.info("Loaded existing script (%d scenes)", len(script["scenes"]))
    else:
        transcripts = load_transcripts()
        script = generate_script(transcripts, zeitgeist)
        review_script(script)

    scenes = script["scenes"]
    narration_text = script["narration"]

    # ── Step 3: Images ──
    if not args.only_stitch and not args.skip_images:
        log.info("=" * 60)
        log.info("STEP 3: Generating images with Gemini")
        log.info("=" * 60)
        refs = load_references()
        for s in scenes:
            sn = s["scene"]
            if sn < args.start_scene or sn > args.end_scene: continue
            result = generate_image(s, refs, sn)
            if not result:
                log.warning("  Retry scene %d...", sn)
                time.sleep(5)
                generate_image(s, refs, sn)

    # ── Step 4: Upload images ──
    image_urls = {}
    if not args.only_stitch and not args.skip_upload:
        log.info("=" * 60)
        log.info("STEP 4: Uploading images to CDN")
        log.info("=" * 60)
        image_urls = upload_images(scenes)
    elif (SCENE_DIR / "image_urls.json").exists():
        image_urls = json.loads((SCENE_DIR / "image_urls.json").read_text())

    # ── Step 5a + 5b: Videos (both pipelines) ──
    if not args.only_stitch:
        log.info("=" * 60)
        log.info("STEP 5: Generating videos (WaveSpeed + Grok)")
        log.info("=" * 60)

        for s in scenes:
            sn = s["scene"]
            if sn < args.start_scene or sn > args.end_scene: continue
            url = image_urls.get(str(sn))
            if not url:
                log.warning("No URL for scene %d, skipping", sn)
                continue

            # WaveSpeed
            if not args.skip_wavespeed:
                try:
                    generate_wavespeed_video(s, sn, url)
                except Exception as e:
                    log.error("  WS scene %d failed: %s", sn, e)

            # Grok
            if not args.skip_grok_video:
                try:
                    generate_grok_video(s, sn, url)
                except Exception as e:
                    log.error("  GK scene %d failed: %s", sn, e)

    # ── Step 6: TTS ──
    narration_audio = None
    if not args.only_stitch and not args.skip_narration:
        log.info("=" * 60)
        log.info("STEP 6: Generating TTS narration")
        log.info("=" * 60)
        narration_audio = generate_narration(narration_text)
    elif (SCENE_DIR / "full_narration.mp3").exists():
        narration_audio = SCENE_DIR / "full_narration.mp3"

    # ── Step 7: Stitch both ──
    log.info("=" * 60)
    log.info("STEP 7: Stitching videos")
    log.info("=" * 60)

    ws_video = stitch("wavespeed", scenes) if not args.skip_wavespeed else None
    gk_video = stitch("grok", scenes) if not args.skip_grok_video else None

    # ── Step 8: Fit narration + overlay ──
    log.info("=" * 60)
    log.info("STEP 8: Fitting narration + overlay")
    log.info("=" * 60)

    ws_final = fit_and_overlay(ws_video, narration_audio, "wavespeed") if ws_video else None
    gk_final = fit_and_overlay(gk_video, narration_audio, "grok") if gk_video else None

    # ── Step 9: Metadata ──
    log.info("=" * 60)
    log.info("STEP 9: Saving metadata")
    log.info("=" * 60)
    meta = save_metadata(script, zeitgeist, ws_final, gk_final)

    # ── Final summary ──
    log.info("")
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE — RUN %s", RUN_ID)
    log.info("")
    if ws_final and ws_final.exists():
        log.info("  WaveSpeed: %s (%.1fs)", ws_final.name, get_duration(ws_final))
    if gk_final and gk_final.exists():
        log.info("  Grok:      %s (%.1fs)", gk_final.name, get_duration(gk_final))
    log.info("")
    log.info("  COSTS:")
    total = 0
    for k, v in sorted(run_costs.items()):
        log.info("    %-20s $%.2f", k, v)
        total += v
    log.info("    %-20s $%.2f", "TOTAL", total)
    log.info("")
    log.info("  ALL FILES: %s", SCENE_DIR)
    log.info("  METADATA:  %s", OUTPUT_DIR / f"metadata_{RUN_ID}.json")
    log.info("  LOG:       %s", log_filename)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
