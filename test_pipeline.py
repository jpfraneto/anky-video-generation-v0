#!/usr/bin/env python3
"""
Test pipeline — runs 1 scene through every step to verify everything works.
Run each step one at a time with the --step flag.

Usage:
  python3 test_pipeline.py --step 1   # Fetch zeitgeist from Grok
  python3 test_pipeline.py --step 2   # Generate script (1 scene + narration) with Claude
  python3 test_pipeline.py --step 3   # Generate image with Gemini (+ 2 random refs)
  python3 test_pipeline.py --step 4   # Upload image to WaveSpeed
  python3 test_pipeline.py --step 5   # Generate video with WaveSpeed wan-2.6-pro
  python3 test_pipeline.py --step 6   # Generate video with Grok grok-imagine-video
  python3 test_pipeline.py --step 7   # Trim WaveSpeed video to 4s with ffmpeg
  python3 test_pipeline.py --step 8   # Generate full narration with TTS
  python3 test_pipeline.py --step 9   # Overlay narration on trimmed video
"""

import os
import sys
import json
import time
import random
import base64
import subprocess
import requests
from pathlib import Path

WAVESPEED_API_KEY = os.environ.get("WAVESPEED_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "")

WAVESPEED_BASE = "https://api.wavespeed.ai/api/v3"
ANTHROPIC_BASE = "https://api.anthropic.com/v1/messages"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROK_BASE = "https://api.x.ai/v1"

REFERENCE_DIR = Path("/home/kithkui/anky/src/public")
BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR / "test_run"
YOUTUBE_DIR = BASE_DIR / "youtube_data"


def download_with_progress(url, dest):
    """Download a file with progress display and 128KB chunks."""
    resp = requests.get(url, stream=True, timeout=(10, 30))
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=131072):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r    Downloading: {pct}% ({downloaded // 1024}KB / {total // 1024}KB)", end="", flush=True)
    if total:
        print()
    return dest


def get_audio_duration(path):
    """Get duration in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    dur = result.stdout.strip()
    return float(dur) if dur else 0.0


def wavespeed_poll(request_id, max_wait=600, interval=5):
    """Poll WaveSpeed for completion."""
    url = f"{WAVESPEED_BASE}/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {WAVESPEED_API_KEY}"}
    start = time.time()
    while time.time() - start < max_wait:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        d = r.json()["data"]
        elapsed = int(time.time() - start)
        print(f"    [{elapsed}s] {d['status']}")
        if d["status"] == "completed":
            return d["outputs"][0]
        if d["status"] == "failed":
            print(f"    FAILED: {d}")
            return None
        time.sleep(interval)
    print(f"    TIMEOUT after {max_wait}s")
    return None


def grok_poll(request_id, max_wait=600, interval=8):
    """Poll Grok for video completion."""
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    start = time.time()
    while time.time() - start < max_wait:
        r = requests.get(f"{GROK_BASE}/videos/{request_id}", headers=headers, timeout=30)
        r.raise_for_status()
        d = r.json()
        elapsed = int(time.time() - start)
        print(f"    [{elapsed}s] {d['status']}")
        if d["status"] == "done":
            return d["video"]["url"]
        if d["status"] == "expired":
            print(f"    EXPIRED: {d}")
            return None
        time.sleep(interval)
    print(f"    TIMEOUT after {max_wait}s")
    return None


# ── Step 1: Grok zeitgeist ──────────────────────────────────────────────────

def step1_zeitgeist():
    """Fetch current cultural zeitgeist from Grok."""
    print("STEP 1: Fetch zeitgeist from Grok (xAI)")
    print("=" * 50)

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "grok-3-mini-fast",
        "messages": [{
            "role": "user",
            "content": (
                "Give me a dense, raw dump of the TOP 10 things the world is talking about "
                "RIGHT NOW on X/Twitter and the internet. Include the dominant cultural mood, "
                "breakthroughs in tech/AI/crypto/consciousness/spirituality, and what feels like "
                "it's about to break through. Be specific — names, events, vibes. ~200 words."
            ),
        }],
        "temperature": 0.8,
        "max_tokens": 600,
    }

    resp = requests.post(f"{GROK_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    zeitgeist = resp.json()["choices"][0]["message"]["content"]

    TEST_DIR.mkdir(exist_ok=True)
    with open(TEST_DIR / "zeitgeist.txt", "w") as f:
        f.write(zeitgeist)

    print(f"OK! Zeitgeist ({len(zeitgeist.split())} words):")
    print(zeitgeist[:500])
    return True


# ── Step 2: Script with Claude ──────────────────────────────────────────────

def step2_script():
    """Generate 1 scene + continuous narration with Claude."""
    print("\nSTEP 2: Generate script with Claude")
    print("=" * 50)

    if not ANTHROPIC_API_KEY:
        print("ERROR: export ANTHROPIC_API_KEY first")
        return False

    # Load zeitgeist if available
    zeitgeist = ""
    zpath = TEST_DIR / "zeitgeist.txt"
    if zpath.exists():
        zeitgeist = zpath.read_text()
        print(f"  Loaded zeitgeist ({len(zeitgeist.split())} words)")

    # Load a small chunk of transcript
    transcripts = []
    for f in sorted(YOUTUBE_DIR.glob("*_transcript.txt")):
        with open(f) as fh:
            transcripts.append(fh.read()[:2000])
        if len(transcripts) >= 2:
            break
    sample = "\n\n".join(transcripts)

    user_content = f"Here is some transcript:\n{sample}\n\n"
    if zeitgeist:
        user_content += f"Current zeitgeist:\n{zeitgeist}\n\n"

    user_content += (
        "Generate a JSON object with:\n"
        '1. "scenes" — array with 1 object: {"scene": 1, "image_prompt": "...", "video_prompt": "...", "grok_duration": 5}\n'
        '   grok_duration = integer 1-15 (seconds for variable-length Grok video).\n'
        '2. "narration" — a continuous ~25 word narration for this one scene about Anky, '
        "a mystical blue-skinned consciousness mirror creature. Poetic, powerful. "
        "Weave in the zeitgeist if provided.\n"
        "Output ONLY the JSON object."
    )

    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": user_content}],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    resp = requests.post(ANTHROPIC_BASE, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"].strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    script = json.loads(text)
    TEST_DIR.mkdir(exist_ok=True)
    with open(TEST_DIR / "test_script.json", "w") as f:
        json.dump(script, f, indent=2)

    print(f"OK! Script saved.")
    print(f"Scene: {json.dumps(script['scenes'][0], indent=2)}")
    print(f"Narration: {script['narration']}")
    return True


# ── Step 3: Image with Gemini ───────────────────────────────────────────────

def step3_image():
    """Generate an image with Gemini using refs + 2 random chaos images."""
    print("\nSTEP 3: Generate image with Gemini")
    print("=" * 50)

    script_path = TEST_DIR / "test_script.json"
    if not script_path.exists():
        print("ERROR: Run step 2 first")
        return False

    with open(script_path) as f:
        script = json.load(f)
    scene = script["scenes"][0]

    # Load canonical refs
    refs = []
    for i in range(1, 4):
        ref_path = REFERENCE_DIR / f"anky-{i}.png"
        if ref_path.exists():
            with open(ref_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            refs.append({"mime_type": "image/png", "data": b64})
            print(f"  Loaded canonical ref: anky-{i}.png")

    # 2 random chaos refs
    all_pngs = [p for p in BASE_DIR.glob("*.png") if not p.name.startswith("prompt_")]
    if len(all_pngs) >= 2:
        chosen = random.sample(all_pngs, 2)
        for p in chosen:
            with open(p, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            refs.append({"mime_type": "image/png", "data": b64})
            print(f"  Loaded random chaos ref: {p.name}")

    prompt = (
        f"Create a mystical fantasy illustration: {scene['image_prompt']}\n\n"
        "CHARACTER - ANKY (follow the reference images exactly):\n"
        "- Blue-skinned creature with large expressive pointed ears\n"
        "- Purple swirling hair with golden spiral accents\n"
        "- Golden/amber glowing eyes, golden jewelry\n"
        "- Compact body, ancient yet childlike\n\n"
        "STYLE: Rich blues/purples/golds, painterly, atmospheric, "
        "slightly psychedelic, cinematic 16:9 composition."
    )

    parts = [{"inline_data": ref} for ref in refs] + [{"text": prompt}]
    payload = {
        "contents": [{"parts": parts}],
        "generation_config": {
            "response_modalities": ["TEXT", "IMAGE"],
            "image_config": {"aspect_ratio": "16:9"},
        },
    }

    url = f"{GEMINI_BASE}/gemini-2.5-flash-image:generateContent?key={GEMINI_API_KEY}"
    print(f"  Calling Gemini with {len(refs)} reference images...")
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} — {resp.text[:500]}")
        return False
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        print(f"ERROR: No candidates. Response: {json.dumps(data)[:300]}")
        return False

    for part in candidates[0].get("content", {}).get("parts", []):
        idata = part.get("inline_data") or part.get("inlineData")
        if idata:
            mime = idata.get("mime_type") or idata.get("mimeType", "")
            if mime.startswith("image/"):
                img_bytes = base64.standard_b64decode(idata["data"])
                img_path = TEST_DIR / "test_scene.png"
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                print(f"OK! Image saved to {img_path} ({len(img_bytes)} bytes)")
                return True

    print("ERROR: No image in Gemini response")
    return False


# ── Step 4: Upload to WaveSpeed ─────────────────────────────────────────────

def step4_upload():
    """Upload the test image to WaveSpeed."""
    print("\nSTEP 4: Upload image to WaveSpeed")
    print("=" * 50)

    img_path = TEST_DIR / "test_scene.png"
    if not img_path.exists():
        print("ERROR: Run step 3 first")
        return False

    headers = {"Authorization": f"Bearer {WAVESPEED_API_KEY}"}
    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{WAVESPEED_BASE}/media/upload/binary",
            headers=headers,
            files={"file": (img_path.name, f, "image/png")},
            timeout=60,
        )
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 200:
        print(f"ERROR: Upload failed: {data}")
        return False

    url = data["data"].get("download_url") or data["data"].get("url")
    with open(TEST_DIR / "test_image_url.txt", "w") as f:
        f.write(url)

    print(f"OK! Uploaded → {url}")
    return True


# ── Step 5: Generate video ──────────────────────────────────────────────────

def step5_video():
    """Generate a video from the uploaded image."""
    print("\nSTEP 5: Generate video with WaveSpeed wan-2.6-pro")
    print("=" * 50)

    url_path = TEST_DIR / "test_image_url.txt"
    script_path = TEST_DIR / "test_script.json"
    if not url_path.exists() or not script_path.exists():
        print("ERROR: Run steps 2-4 first")
        return False

    image_url = url_path.read_text().strip()
    with open(script_path) as f:
        script = json.load(f)
    video_prompt = script["scenes"][0]["video_prompt"]

    headers = {
        "Authorization": f"Bearer {WAVESPEED_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "image": image_url,
        "prompt": video_prompt,
        "resolution": "1080p",
        "duration": 5,
        "seed": -1,
    }

    print(f"  Prompt: {video_prompt}")
    print(f"  Submitting to wan-2.6-pro (1080p, 5s)...")
    resp = requests.post(
        f"{WAVESPEED_BASE}/alibaba/wan-2.6/image-to-video-pro",
        headers=headers, json=payload, timeout=60,
    )
    resp.raise_for_status()
    request_id = resp.json()["data"]["id"]
    print(f"  Job submitted: {request_id}")

    print("  Polling for result...")
    video_url = wavespeed_poll(request_id, max_wait=600, interval=8)
    if not video_url:
        return False

    video_path = TEST_DIR / "test_scene.mp4"
    download_with_progress(video_url, video_path)
    sz = video_path.stat().st_size
    print(f"OK! Video saved to {video_path} ({sz // 1024}KB)")
    return True


# ── Step 6: Grok video ──────────────────────────────────────────────────────

def step6_grok_video():
    """Generate a video with Grok grok-imagine-video from the uploaded image."""
    print("\nSTEP 6: Generate video with Grok grok-imagine-video")
    print("=" * 50)

    url_path = TEST_DIR / "test_image_url.txt"
    script_path = TEST_DIR / "test_script.json"
    if not url_path.exists() or not script_path.exists():
        print("ERROR: Run steps 2-4 first")
        return False

    image_url = url_path.read_text().strip()
    with open(script_path) as f:
        script = json.load(f)
    video_prompt = script["scenes"][0]["video_prompt"]
    grok_duration = script["scenes"][0].get("grok_duration", 5)

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "grok-imagine-video",
        "prompt": video_prompt,
        "image_url": image_url,
        "duration": grok_duration,
        "aspect_ratio": "16:9",
        "resolution": "720p",
    }

    print(f"  Prompt: {video_prompt}")
    print(f"  Duration: {grok_duration}s")
    print(f"  Image URL: {image_url[:80]}...")
    print(f"  Submitting to grok-imagine-video...")
    resp = requests.post(f"{GROK_BASE}/videos/generations", headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} — {resp.text[:500]}")
        return False

    data = resp.json()
    request_id = data.get("request_id")
    if not request_id:
        print(f"ERROR: No request_id in response: {data}")
        return False
    print(f"  Job submitted: {request_id}")

    print("  Polling for result...")
    video_url = grok_poll(request_id, max_wait=600, interval=8)
    if not video_url:
        return False

    video_path = TEST_DIR / "test_scene_grok.mp4"
    download_with_progress(video_url, video_path)
    sz = video_path.stat().st_size
    dur = get_audio_duration(video_path)
    if dur > 0:
        print(f"OK! Grok video saved to {video_path} ({sz // 1024}KB, {dur:.1f}s)")
    else:
        print(f"OK! Grok video saved to {video_path} ({sz // 1024}KB)")
    return True


# ── Step 7: Trim video ──────────────────────────────────────────────────────

def step7_trim():
    """Trim the 5s WaveSpeed video to 4s."""
    print("\nSTEP 7: Trim WaveSpeed video to 4s with ffmpeg")
    print("=" * 50)

    src = TEST_DIR / "test_scene.mp4"
    dst = TEST_DIR / "test_scene_trimmed.mp4"
    if not src.exists():
        print("ERROR: Run step 5 first")
        return False

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-t", "4", "-c", "copy", str(dst)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  Stream copy failed, re-encoding...")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-t", "4",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an", str(dst)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: {result.stderr[-300:]}")
            return False

    dur = get_audio_duration(dst)
    if dur > 0:
        print(f"OK! Trimmed video: {dst} ({dur:.1f}s)")
    else:
        sz = dst.stat().st_size if dst.exists() else 0
        print(f"OK! Trimmed video: {dst} ({sz // 1024}KB)")
    return True


# ── Step 8: TTS narration ───────────────────────────────────────────────────

def step8_tts():
    """Generate narration with WaveSpeed vibevoice TTS."""
    print("\nSTEP 8: Generate narration with WaveSpeed TTS (vibevoice)")
    print("=" * 50)

    script_path = TEST_DIR / "test_script.json"
    if not script_path.exists():
        print("ERROR: Run step 2 first")
        return False

    with open(script_path) as f:
        script = json.load(f)
    narration = script.get("narration", "")
    if not narration:
        print("ERROR: No narration in script")
        return False

    print(f'  Narration: "{narration}"')
    print(f"  Words: {len(narration.split())}")

    headers = {
        "Authorization": f"Bearer {WAVESPEED_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"text": narration, "speaker": "Frank"}

    print("  Submitting to vibevoice...")
    resp = requests.post(
        f"{WAVESPEED_BASE}/wavespeed-ai/vibevoice",
        headers=headers, json=payload, timeout=60,
    )
    resp.raise_for_status()
    request_id = resp.json()["data"]["id"]
    print(f"  Job submitted: {request_id}")

    print("  Polling...")
    audio_url = wavespeed_poll(request_id, max_wait=600, interval=5)
    if not audio_url:
        return False

    audio_path = TEST_DIR / "test_narration.mp3"
    download_with_progress(audio_url, audio_path)
    dur = get_audio_duration(audio_path)
    sz = audio_path.stat().st_size
    print(f"OK! Narration saved: {audio_path} ({sz // 1024}KB, {dur:.1f}s)")
    return True


# ── Step 9: Overlay narration ───────────────────────────────────────────────

def step9_overlay():
    """Time-fit narration and overlay on video."""
    print("\nSTEP 9: Overlay narration on video")
    print("=" * 50)

    video_path = TEST_DIR / "test_scene_trimmed.mp4"
    audio_path = TEST_DIR / "test_narration.mp3"

    if not video_path.exists():
        print("ERROR: Run step 7 first")
        return False
    if not audio_path.exists():
        print("ERROR: Run step 8 first")
        return False

    video_dur = get_audio_duration(video_path)
    audio_dur = get_audio_duration(audio_path)
    print(f"  Video: {video_dur:.1f}s, Audio: {audio_dur:.1f}s")

    # Time-fit narration to match video
    fitted_path = TEST_DIR / "test_narration_fitted.mp3"
    if audio_dur > 0 and video_dur > 0:
        ratio = audio_dur / video_dur
        print(f"  Tempo ratio: {ratio:.3f}")
        if abs(ratio - 1.0) > 0.02:
            # Build atempo filter chain (0.5-2.0 range)
            filters = []
            r = ratio
            while r > 2.0:
                filters.append("atempo=2.0")
                r /= 2.0
            while r < 0.5:
                filters.append("atempo=0.5")
                r *= 2.0
            filters.append(f"atempo={r:.4f}")
            filter_str = ",".join(filters)
            print(f"  Applying tempo: {filter_str}")

            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path),
                 "-filter:a", filter_str,
                 "-c:a", "libmp3lame", "-b:a", "192k",
                 str(fitted_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  Tempo failed, using original: {result.stderr[-200:]}")
                fitted_path = audio_path
            else:
                new_dur = get_audio_duration(fitted_path)
                print(f"  Fitted: {audio_dur:.1f}s → {new_dur:.1f}s")
        else:
            print("  Audio already fits, no adjustment needed")
            fitted_path = audio_path
    else:
        fitted_path = audio_path

    # Overlay
    output = TEST_DIR / "test_scene_narrated.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(video_path),
         "-i", str(fitted_path),
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k",
         "-shortest",
         str(output)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print(f"  Copy failed, re-encoding...")
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(video_path),
             "-i", str(fitted_path),
             "-map", "0:v", "-map", "1:a",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", "-b:a", "192k",
             "-shortest",
             str(output)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: {result.stderr[-300:]}")
            return False

    dur = get_audio_duration(output)
    if dur > 0:
        print(f"OK! Narrated video: {output} ({dur:.1f}s)")
    else:
        sz = output.stat().st_size
        print(f"OK! Narrated video: {output} ({sz // 1024}KB)")

    print()
    print("=" * 50)
    print("ALL STEPS PASSED!")
    print(f"  Test files in: {TEST_DIR}/")
    print(f"  Cost: ~$0.86 (Grok chat ~$0.01 + Grok video ~$0.25 + Gemini $0.04 + WaveSpeed $0.60 + TTS ~$0.01)")
    print("=" * 50)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test pipeline — 1 scene through every step")
    parser.add_argument("--step", type=int, required=True, choices=range(1, 10),
                        help="Which step to run (1-9)")
    args = parser.parse_args()

    steps = {
        1: step1_zeitgeist,
        2: step2_script,
        3: step3_image,
        4: step4_upload,
        5: step5_video,
        6: step6_grok_video,
        7: step7_trim,
        8: step8_tts,
        9: step9_overlay,
    }

    ok = steps[args.step]()
    sys.exit(0 if ok else 1)
