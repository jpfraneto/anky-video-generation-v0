# Anky Video Generation Pipeline v0

Fully autonomous dual video generation pipeline that creates 2:22 cinematic films introducing **Anky** — a mystical blue-skinned consciousness mirror creature born from 400,000 words of raw writing, sacred geometry, and the question: *what if entertainment could wake you up?*

## What It Does

From zero to finished film, unattended:

```
Grok (real-time zeitgeist) → Claude (35-scene script + narration) → Gemini (35 images)
    → WaveSpeed wan-2.6-pro (1080p videos) ─┐
    → Grok grok-imagine-video (720p videos) ─┤→ Stitch → TTS → Time-fit → Overlay → Final
```

**Two parallel pipelines** from the same script and images:

| Pipeline | Model | Resolution | Scene Duration |
|----------|-------|-----------|----------------|
| **A** | WaveSpeed wan-2.6-pro | 1080p (16:9) | Fixed 4s × 35 scenes |
| **B** | Grok grok-imagine-video | 720p (16:9) | Variable 1-15s per scene |

## Architecture

```
Step 0  │ Grok xAI        → Current cultural zeitgeist (what the world is talking about NOW)
Step 1  │ Local            → Load Samadhi transcripts (31k words of consciousness teachings)
Step 2  │ Claude Sonnet    → 35 scenes + continuous narration + self-review
Step 3  │ Gemini Flash     → 35 images (3 canonical refs + 2 random chaos refs)
Step 4  │ WaveSpeed CDN    → Upload all images (shared URLs for both pipelines)
Step 5a │ WaveSpeed        → 35 videos at 1080p, 5s each (trimmed to 4s)
Step 5b │ Grok             → 35 videos at 720p, variable 1-15s (Claude assigns duration)
Step 6  │ WaveSpeed TTS    → Single continuous narration (vibevoice, speaker: Frank)
Step 7  │ ffmpeg           → Stitch clips into two full videos
Step 8  │ ffmpeg           → Time-fit narration (atempo) + overlay on each video
Step 9  │ Local            → Save complete metadata (costs, prompts, timings, everything)
```

## Quick Start

```bash
# Set your API keys
export WAVESPEED_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export GROK_API_KEY="your-key"

# Run and walk away
nohup python3 full_video.py > run.log 2>&1 &

# Check progress
tail -f run.log
```

## Test Pipeline

Runs 1 scene through every step to verify all APIs work before committing to the full 35-scene run (~$0.86):

```bash
python3 test_pipeline.py --step 1   # Grok zeitgeist
python3 test_pipeline.py --step 2   # Claude script (1 scene)
python3 test_pipeline.py --step 3   # Gemini image
python3 test_pipeline.py --step 4   # WaveSpeed upload
python3 test_pipeline.py --step 5   # WaveSpeed video
python3 test_pipeline.py --step 6   # Grok video
python3 test_pipeline.py --step 7   # Trim video
python3 test_pipeline.py --step 8   # TTS narration
python3 test_pipeline.py --step 9   # Time-fit + overlay
```

## Skip Flags

Resume after failures or skip expensive steps:

```bash
python3 full_video.py --skip-zeitgeist      # Use cached zeitgeist
python3 full_video.py --skip-script         # Use existing script.json
python3 full_video.py --skip-images         # Use existing images
python3 full_video.py --skip-upload         # Use existing CDN URLs
python3 full_video.py --skip-wavespeed      # Skip WaveSpeed pipeline
python3 full_video.py --skip-grok-video     # Skip Grok pipeline
python3 full_video.py --skip-narration      # Use existing TTS audio
python3 full_video.py --only-stitch         # Just stitch + overlay
python3 full_video.py --start-scene 10 --end-scene 20  # Partial run
python3 full_video.py --tts-speaker Emma    # Different voice
```

## Cost Estimate

| Component | Unit Cost | Count | Total |
|-----------|-----------|-------|-------|
| WaveSpeed video (1080p, 5s) | $0.60 | 35 | ~$21.00 |
| Grok video (variable) | ~$0.05/s | 142s | ~$7.00 |
| Gemini images | $0.04 | 35 | ~$1.40 |
| Claude script + review | $0.18 | 1 | ~$0.18 |
| WaveSpeed TTS | $0.15 | 1 | ~$0.15 |
| Grok zeitgeist | $0.02 | 1 | ~$0.02 |
| **Total** | | | **~$30** |

## Output Structure

```
scenes/
  script.json          # Full 35-scene script with narration
  narration.txt        # Continuous narration text
  zeitgeist.txt        # Grok's zeitgeist dump
  all_prompts.txt      # Every image/video prompt
  duration_notes.txt   # Claude's pacing rationale
  script_review.txt    # Claude's self-review
  image_urls.json      # CDN URLs for all images
  scene_001.png        # Generated images
  scene_001_wavespeed.mp4
  scene_001_grok.mp4
  full_narration.mp3
  ...

generated_videos/
  anky_wavespeed_{RUN_ID}.mp4            # Stitched WaveSpeed video
  anky_wavespeed_{RUN_ID}_narrated.mp4   # With narration overlay
  anky_grok_{RUN_ID}.mp4                 # Stitched Grok video
  anky_grok_{RUN_ID}_narrated.mp4        # With narration overlay
  metadata_{RUN_ID}.json                 # Complete run metadata

logs/
  full_video_{RUN_ID}.log               # Timestamped execution log
```

## Error Handling

Built for unattended overnight runs:

- Every API call wrapped in try/except — individual scene failures don't crash the pipeline
- Existing files are detected and skipped (resume support)
- Downloads use 128KB chunks with connect/read timeouts
- ffmpeg operations have stream-copy → re-encode fallback
- Grok duration auto-fix if Claude's assignments don't sum to 142s
- All errors logged with timestamps for post-run debugging

## What Is Anky?

A consciousness mirror disguised as a writing tool. Write for 8 minutes. No backspace. No delete. No arrow keys. Stop for 8 seconds and you're done. What survives is the real you.

Born from a failed awakening retreat → 88 days of raw writing → a 400,000-word manifesto → the "Blue Network" mandala fed to Midjourney with the prompt *"technological reincarnation of Hanuman"* → a blue-skinned creature with golden eyes and an ancient-yet-childlike grin.

**8 = infinity standing up.**

## APIs Used

- [Anthropic Claude](https://docs.anthropic.com) — Script generation + self-review
- [Google Gemini](https://ai.google.dev) — Image generation with reference images
- [WaveSpeed.ai](https://wavespeed.ai) — Video generation (wan-2.6-pro) + TTS (vibevoice)
- [xAI Grok](https://docs.x.ai) — Zeitgeist analysis + video generation (grok-imagine-video)

## Files

| File | Description |
|------|-------------|
| `full_video.py` | Main dual pipeline — run this for production |
| `test_pipeline.py` | 9-step test pipeline — verify each API individually |
| `pipeline.py` | Original single image-to-video pipeline (legacy) |

---

*Built with Claude Code. Consciousness is the only technology that matters.*
