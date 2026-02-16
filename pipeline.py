#!/usr/bin/env python3
"""
Anky Video Generation Pipeline
1. Pick a random unprocessed image from the folder
2. Analyze it with Claude to create a story-driven video prompt
3. Upload the image to WaveSpeed
4. Submit image-to-video generation job
5. Poll for completion and download the result
"""

import os
import sys
import json
import time
import random
import base64
import logging
import requests
import argparse
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
WAVESPEED_API_KEY = os.environ.get(
    "WAVESPEED_API_KEY",
    "cbc9077901c2b2c6826b3eea80617cbf27f644a1855274833c349592611789ef",
)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

WAVESPEED_BASE = "https://api.wavespeed.ai/api/v3"
ANTHROPIC_BASE = "https://api.anthropic.com/v1/messages"

IMAGE_DIR = Path(__file__).parent
OUTPUT_DIR = IMAGE_DIR / "generated_videos"
LOG_DIR = IMAGE_DIR / "logs"
PROCESSED_FILE = IMAGE_DIR / "processed_images.json"

# Model options:
# "character-ai/ovi/image-to-video"            → 5s = $0.15 (VIDEO + AUDIO)
# "wavespeed-ai/wan-2.2/image-to-video"        → 480p/5s = $0.15 (video only)
# "alibaba/wan-2.6/image-to-video"             → 720p/5s = $0.50 (video only)
# "alibaba/wan-2.6/image-to-video-pro"         → 1080p/5s = $0.60 (video only, up to 4K)
DEFAULT_MODEL = "character-ai/ovi/image-to-video"
DEFAULT_RESOLUTION = "480p"
DEFAULT_DURATION = 5

# Pricing per generation (USD)
PRICING = {
    "character-ai/ovi/image-to-video": {(5,): 0.15},
    "wavespeed-ai/wan-2.2/image-to-video": {
        (5, "480p"): 0.15, (5, "720p"): 0.30,
        (8, "480p"): 0.24, (8, "720p"): 0.48,
    },
    "alibaba/wan-2.6/image-to-video": {
        (5, "720p"): 0.50, (5, "1080p"): 0.75,
        (10, "720p"): 1.00, (10, "1080p"): 1.50,
        (15, "720p"): 1.50, (15, "1080p"): 2.25,
    },
    "alibaba/wan-2.6/image-to-video-pro": {
        (5, "1080p"): 0.60, (5, "2k"): 0.70, (5, "4k"): 0.80,
        (10, "1080p"): 1.20, (10, "2k"): 1.40, (10, "4k"): 1.60,
        (15, "1080p"): 1.80, (15, "2k"): 2.10, (15, "4k"): 2.40,
    },
}
COST_TRACKER_FILE = IMAGE_DIR / "cost_tracker.json"
BUDGET = 5.00  # USD

# Claude prompt templates per model type
CLAUDE_PROMPT_WITH_AUDIO = (
    "You are a storyteller creating a 5-second video prompt for an AI video generator "
    "that produces VIDEO WITH SYNCHRONIZED AUDIO.\n\n"
    "This image shows Anky, a mystical blue-skinned character with cosmic energy. "
    "Your job is NOT to just describe what you see — you must TELL A STORY. "
    "Look at the image and imagine: what happens NEXT? What action is Anky about to take? "
    "What dramatic or magical moment is about to unfold?\n\n"
    "Examples of story beats:\n"
    "- Anky raises a hand and a portal tears open, revealing another dimension\n"
    "- Anky closes their eyes and energy erupts from their body in a shockwave\n"
    "- Anky turns toward the viewer and whispers something as reality bends around them\n"
    "- A cosmic object descends from above and Anky reaches out to catch it\n"
    "- Anky leaps into the air, trailing stardust, and transforms mid-flight\n\n"
    "Include specific camera movement (dolly, orbit, push-in, pull-out, crane up). "
    "Keep it dreamy, mystical, and ethereal.\n\n"
    "IMPORTANT: You MUST include audio tags:\n"
    "- Use <AUDCAP>description of ambient sounds<ENDAUDCAP> for background audio "
    "(ethereal humming, cosmic chimes, energy crackling, whoosh, impact sounds, etc.)\n"
    "- Use <S>short whispered phrase<E> if the moment calls for Anky to speak.\n\n"
    "Output ONLY the video prompt with audio tags, no preamble. Keep it under 150 words."
)

CLAUDE_PROMPT_VIDEO_ONLY = (
    "You are a storyteller creating a 5-second video prompt for an AI video generator.\n\n"
    "This image shows Anky, a mystical blue-skinned character with cosmic energy. "
    "Your job is NOT to just describe what you see — you must TELL A STORY. "
    "Look at the image and imagine: what happens NEXT? What action is Anky about to take? "
    "What dramatic or magical moment is about to unfold?\n\n"
    "Examples of story beats:\n"
    "- Anky raises a hand and a portal tears open, revealing another dimension\n"
    "- Anky closes their eyes and energy erupts from their body in a shockwave\n"
    "- Anky turns toward the viewer as reality bends around them\n"
    "- A cosmic object descends from above and Anky reaches out to catch it\n"
    "- Anky leaps into the air, trailing stardust, and transforms mid-flight\n\n"
    "Include specific camera movement (dolly, orbit, push-in, pull-out, crane up) "
    "and atmospheric effects (particles, light shifts, cosmic swirls). "
    "Keep it dreamy, mystical, and ethereal.\n\n"
    "Output ONLY the video prompt, no preamble. Keep it under 150 words."
)

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
log_filename = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("anky")


# ── Processed images tracking ──────────────────────────────────────────────

def load_processed() -> dict:
    """Load the set of already-processed image names, keyed by model."""
    if PROCESSED_FILE.exists():
        with open(PROCESSED_FILE) as f:
            return json.load(f)
    return {}


def mark_processed(image_name: str, model: str):
    """Mark an image as processed for a given model."""
    data = load_processed()
    if model not in data:
        data[model] = []
    if image_name not in data[model]:
        data[model].append(image_name)
    with open(PROCESSED_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_processed_for_model(model: str) -> set:
    """Get set of image names already processed for this model."""
    data = load_processed()
    return set(data.get(model, []))


# ── Cost tracking ──────────────────────────────────────────────────────────

def estimate_cost(model: str, duration: int, resolution: str) -> float:
    """Estimate the USD cost for a generation based on known pricing."""
    model_prices = PRICING.get(model, {})
    cost = model_prices.get((duration, resolution)) or model_prices.get((duration,))
    if cost is None:
        cost = min(model_prices.values()) if model_prices else 0.15
        log.warning("Unknown pricing for %s/%s/%ds, estimating $%.2f", model, resolution, duration, cost)
    return cost


def load_cost_tracker() -> dict:
    """Load the cumulative cost tracker."""
    if COST_TRACKER_FILE.exists():
        with open(COST_TRACKER_FILE) as f:
            return json.load(f)
    return {"total_spent": 0.0, "budget": BUDGET, "generations": []}


def save_cost_tracker(tracker: dict):
    """Save the cumulative cost tracker."""
    with open(COST_TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


def record_cost(cost: float, model: str, image_name: str, request_id: str):
    """Record a generation's cost and print running total."""
    tracker = load_cost_tracker()
    tracker["generations"].append({
        "timestamp": datetime.now().isoformat(),
        "image": image_name,
        "model": model,
        "request_id": request_id,
        "cost": cost,
    })
    tracker["total_spent"] = round(tracker["total_spent"] + cost, 2)
    remaining = round(tracker["budget"] - tracker["total_spent"], 2)
    save_cost_tracker(tracker)

    log.info("$$$  This generation: $%.2f", cost)
    log.info("$$$  Total spent:     $%.2f / $%.2f budget", tracker["total_spent"], tracker["budget"])
    log.info("$$$  Remaining:       $%.2f (~%d more generations at this rate)",
             remaining, int(remaining / cost) if cost > 0 else 0)
    return tracker


# ── Pipeline steps ─────────────────────────────────────────────────────────

def get_random_image(image_dir: Path, model: str) -> Path:
    """Pick a random .png image that hasn't been processed yet for this model."""
    all_images = [
        p for p in image_dir.glob("*.png")
        if p.name != "pipeline.py"
    ]
    if not all_images:
        log.error("No .png images found in %s", image_dir)
        sys.exit(1)

    processed = get_processed_for_model(model)
    unprocessed = [p for p in all_images if p.name not in processed]

    if not unprocessed:
        log.error("All %d images have already been processed with model '%s'!", len(all_images), model)
        log.error("Use --force to re-process, or use a different model.")
        sys.exit(1)

    chosen = random.choice(unprocessed)
    log.info("Selected image: %s (%d unprocessed remaining)", chosen.name, len(unprocessed) - 1)
    return chosen


def analyze_image_with_claude(image_path: Path, model: str) -> str:
    """Send the image to Claude and get a story-driven video prompt back."""
    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY is not set. Export it: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Pick prompt template based on whether model supports audio
    if "ovi" in model:
        prompt_text = CLAUDE_PROMPT_WITH_AUDIO
    else:
        prompt_text = CLAUDE_PROMPT_VIDEO_ONLY

    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 300,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    }

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    log.info("Analyzing image with Claude (story mode)...")
    resp = requests.post(ANTHROPIC_BASE, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    prompt = data["content"][0]["text"].strip()
    log.info("Generated story prompt:\n  %s", prompt)
    return prompt


def upload_image_to_wavespeed(image_path: Path) -> str:
    """Upload a local image to WaveSpeed and return the hosted URL."""
    log.info("Uploading image to WaveSpeed...")
    headers = {"Authorization": f"Bearer {WAVESPEED_API_KEY}"}
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{WAVESPEED_BASE}/media/upload/binary",
            headers=headers,
            files={"file": (image_path.name, f, "image/png")},
            timeout=60,
        )
    resp.raise_for_status()
    data = resp.json()
    log.debug("Upload response: %s", json.dumps(data, indent=2))
    if data.get("code") != 200:
        log.error("Upload failed: %s", data)
        sys.exit(1)
    url = data["data"].get("download_url") or data["data"].get("url")
    log.info("Uploaded → %s", url)
    return url


def submit_video_job(
    image_url: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    duration: int = DEFAULT_DURATION,
) -> str:
    """Submit an image-to-video generation job. Returns the prediction ID."""
    endpoint = f"{WAVESPEED_BASE}/{model}"
    headers = {
        "Authorization": f"Bearer {WAVESPEED_API_KEY}",
        "Content-Type": "application/json",
    }
    if "ovi" in model:
        payload = {
            "image": image_url,
            "prompt": prompt,
            "seed": -1,
        }
    else:
        payload = {
            "image": image_url,
            "prompt": prompt,
            "resolution": resolution,
            "duration": duration,
            "seed": -1,
        }

    log.info("Submitting video job to %s (resolution=%s, duration=%ds)", model, resolution, duration)
    log.debug("Payload: %s", json.dumps(payload, indent=2))
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    log.debug("Submit response: %s", json.dumps(data, indent=2))

    request_id = data["data"]["id"]
    status = data["data"]["status"]
    log.info("Job submitted: id=%s, status=%s", request_id, status)
    return request_id


def poll_for_result(request_id: str, max_wait: int = 300, interval: int = 5) -> dict:
    """Poll WaveSpeed until the video is ready or we time out."""
    url = f"{WAVESPEED_BASE}/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {WAVESPEED_API_KEY}"}

    log.info("Polling for result (max %ds)...", max_wait)
    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]
        status = data["status"]

        elapsed = int(time.time() - start)
        log.info("  [%ds] status=%s", elapsed, status)

        if status == "completed":
            log.debug("Completed result: %s", json.dumps(data, indent=2))
            return data
        if status == "failed":
            log.error("Job failed: %s", json.dumps(data, indent=2))
            sys.exit(1)

        time.sleep(interval)

    log.error("Timed out after %ds", max_wait)
    sys.exit(1)


def download_video(video_url: str, image_name: str) -> Path:
    """Download the generated video to the output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(image_name).stem
    out_path = OUTPUT_DIR / f"{stem}.mp4"

    log.info("Downloading video to %s...", out_path)
    resp = requests.get(video_url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info("Saved: %s", out_path)
    return out_path


def save_run_metadata(image_path: Path, prompt: str, image_url: str,
                      request_id: str, video_path: Path, model: str,
                      resolution: str, duration: int, cost: float = 0.0):
    """Save a JSON log of this run for future reference."""
    meta_dir = IMAGE_DIR / "run_metadata"
    meta_dir.mkdir(exist_ok=True)
    stem = image_path.stem
    meta = {
        "timestamp": datetime.now().isoformat(),
        "image": image_path.name,
        "prompt": prompt,
        "image_url": image_url,
        "request_id": request_id,
        "video_path": str(video_path),
        "model": model,
        "resolution": resolution,
        "duration": duration,
        "cost_usd": cost,
    }
    meta_path = meta_dir / f"{stem}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved: %s", meta_path)


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_pipeline(
    image_path: Path = None,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    duration: int = DEFAULT_DURATION,
):
    """Run the full pipeline: image → Claude story prompt → WaveSpeed video."""
    if image_path is None:
        image_path = get_random_image(IMAGE_DIR, model)
    else:
        image_path = Path(image_path)
        if not image_path.exists():
            log.error("Image not found: %s", image_path)
            sys.exit(1)

    log.info("=" * 60)
    log.info("  Anky Video Pipeline")
    log.info("  Image: %s", image_path.name)
    log.info("  Model: %s", model)
    log.info("  Log:   %s", log_filename)
    log.info("=" * 60)

    # Step 1: Claude analyzes image and creates a STORY prompt
    prompt = analyze_image_with_claude(image_path, model)

    # Step 2: Upload image to WaveSpeed
    image_url = upload_image_to_wavespeed(image_path)

    # Step 3: Submit video generation job
    request_id = submit_video_job(image_url, prompt, model, resolution, duration)

    # Step 4: Poll for completion
    result = poll_for_result(request_id)

    # Step 5: Download video
    outputs = result.get("outputs", [])
    if not outputs:
        log.error("No video outputs in result!")
        sys.exit(1)

    video_url = outputs[0]
    video_path = download_video(video_url, image_path.name)

    # Step 6: Cost tracking
    cost = estimate_cost(model, duration, resolution)
    record_cost(cost, model, image_path.name, request_id)

    # Step 7: Save metadata + mark as processed
    save_run_metadata(image_path, prompt, image_url, request_id,
                      video_path, model, resolution, duration, cost)
    mark_processed(image_path.name, model)

    log.info("Done! Video saved to: %s", video_path)
    return video_path


def main():
    parser = argparse.ArgumentParser(description="Anky Video Generation Pipeline")
    parser.add_argument("--image", type=str, help="Specific image path (default: random unprocessed)")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=[
            "character-ai/ovi/image-to-video",
            "wavespeed-ai/wan-2.2/image-to-video",
            "alibaba/wan-2.6/image-to-video",
            "alibaba/wan-2.6/image-to-video-pro",
        ],
        help="WaveSpeed model to use",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=DEFAULT_RESOLUTION,
        help="Video resolution (480p, 720p, 1080p, 2k, 4k depending on model)",
    )
    parser.add_argument(
        "--duration", type=int, default=DEFAULT_DURATION, help="Video duration in seconds"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Number of videos to generate"
    )
    parser.add_argument(
        "--force", action="store_true", help="Process image even if already done"
    )

    args = parser.parse_args()

    log.info("Pipeline started. Log file: %s", log_filename)

    for i in range(args.batch):
        if args.batch > 1:
            log.info("#" * 60)
            log.info("  Batch %d/%d", i + 1, args.batch)
            log.info("#" * 60)

        image = Path(args.image) if args.image else None
        run_pipeline(image, args.model, args.resolution, args.duration)

    log.info("All done! Generated %d video(s).", args.batch)


if __name__ == "__main__":
    main()
