import os
import uuid
import math
import re
import shutil
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, url_for
from gtts import gTTS
from PIL import Image
from moviepy.editor import (
    ImageClip, AudioFileClip, VideoFileClip, concatenate_videoclips,
    CompositeAudioClip, afx, vfx
)
import requests

# ------------------ CONFIG ------------------ #
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
IMAGES_DIR = STATIC_DIR / "images"
VIDEOS_DIR = STATIC_DIR / "videos"
MUSIC_DIR = STATIC_DIR / "music"

for d in [STATIC_DIR, IMAGES_DIR, VIDEOS_DIR, MUSIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Optional: Set your Pixabay key here or as an environment variable in Render
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()

# Flask app
app = Flask(__name__)

# ------------------ HELPERS ------------------ #
def split_into_scenes(text: str):
    """
    Split the script into scenes.
    Strategy:
      1) Split by blank lines (paragraphs)
      2) If only one paragraph, split by sentences
    """
    text = text.strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    # fallback: sentence split (simple)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    # group sentences into chunks of 2–3 to form scenes
    scenes = []
    chunk = []
    for s in sentences:
        chunk.append(s)
        if len(chunk) >= 3:
            scenes.append(" ".join(chunk))
            chunk = []
    if chunk:
        scenes.append(" ".join(chunk))

    return scenes if scenes else [text]


def fetch_image_for_query(query: str, out_dir: Path) -> Path:
    """
    Try Pixabay first (if API key present). If not, use Unsplash source (no API).
    Save to out_dir and return the local image path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{uuid.uuid4().hex}.jpg"

    # Try Pixabay
    if PIXABAY_API_KEY:
        try:
            r = requests.get(
                "https://pixabay.com/api/",
                params={
                    "key": PIXABAY_API_KEY,
                    "q": query,
                    "image_type": "photo",
                    "per_page": 3,
                    "safesearch": "true",
                    "orientation": "horizontal"
                },
                timeout=15
            )
            r.raise_for_status()
            data = r.json()
            hits = data.get("hits", [])
            if hits:
                url = hits[0].get("largeImageURL") or hits[0].get("webformatURL")
                if url:
                    img_data = requests.get(url, timeout=15).content
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    return img_path
        except Exception:
            pass

    # Fallback Unsplash Source (random, no API)
    try:
        # “query” affects the random image chosen
        url = f"https://source.unsplash.com/1280x720/?{requests.utils.quote(query)}"
        img_data = requests.get(url, timeout=15).content
        with open(img_path, "wb") as f:
            f.write(img_data)
        return img_path
    except Exception:
        # last fallback: a placeholder
        url = "https://via.placeholder.com/1280x720.png?text=AI+Video"
        img_data = requests.get(url, timeout=15).content
        with open(img_path, "wb") as f:
            f.write(img_data)
        return img_path


def normalize_image_to_size(img_path: Path, size: tuple[int, int]) -> Path:
    """
    Resize/crop the image to exact size while preserving aspect ratio (cover).
    """
    w, h = size
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im_ratio = im.width / im.height
        target_ratio = w / h

        if im_ratio > target_ratio:
            # image too wide -> fit height, crop width
            new_height = h
            new_width = int(im_ratio * new_height)
        else:
            # image too tall -> fit width, crop height
            new_width = w
            new_height = int(new_width / im_ratio)

        im = im.resize((new_width, new_height), Image.LANCZOS)
        left = (new_width - w) // 2
        top = (new_height - h) // 2
        im = im.crop((left, top, left + w, top + h))

        out = img_path.with_suffix(".resized.jpg")
        im.save(out, quality=90)
        return out


def make_ken_burns_clip(image_path: Path, duration: float, size: tuple[int, int], zoom=1.08):
    """
    Create a subtle Ken Burns zoom effect clip for a single image.
    """
    w, h = size
    clip = ImageClip(str(image_path)).set_duration(duration).resize(newsize=size)

    # Zoom effect: slowly scale up from 1.0 to zoom (e.g., 1.08)
    # Implement by resizing over time.
    def zoom_func(get_frame, t):
        frame = get_frame(t)
        # linear zoom over time
        scale = 1.0 + (zoom - 1.0) * (t / max(duration, 0.001))
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # center crop to original size
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img = img.crop((left, top, left + w, top + h))
        return img

    return clip.fl(zoom_func, apply_to=['mask'])


def tts_to_mp3(text: str, out_path: Path, lang="en", speed=1.0):
    """
    Generate speech via gTTS and (optionally) adjust speed using moviepy.
    """
    tmp_mp3 = out_path.with_suffix(".tmp.mp3")
    tts = gTTS(text=text, lang=lang)
    tts.save(tmp_mp3)

    # adjust speed using moviepy
    audio = AudioFileClip(str(tmp_mp3))
    if abs(speed - 1.0) > 0.01:
        audio = afx.audio_speedx(audio, factor=speed)
    audio.write_audiofile(str(out_path), fps=44100, codec="aac", verbose=False, logger=None)
    audio.close()
    try:
        tmp_mp3.unlink(missing_ok=True)
    except Exception:
        pass


# ------------------ ROUTES ------------------ #
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # form inputs
        script = (request.form.get("script") or "").strip()
        topic = (request.form.get("topic") or "").strip()
        mode = request.form.get("mode") # "script" or "topic"
        layout = request.form.get("layout", "landscape") # "landscape" or "vertical"
        music_url = (request.form.get("music_url") or "").strip()
        voice_speed = float(request.form.get("voice_speed", "1.0"))

        # sizes
        if layout == "vertical":
            size = (1080, 1920) # Shorts
        else:
            size = (1280, 720) # YouTube

        # if user selected "topic" mode but left script empty, make a simple template
        if mode == "topic" and not script:
            # VERY simple templated expansion (no paid AI needed)
            script = (
                f"Welcome to our video about {topic}! "
                f"In this video, we'll explore key facts, benefits, and insights about {topic}. "
                f"Let's dive in.\n\n"
                f"Overview:\n"
                f"- What is {topic}?\n"
                f"- Why it matters\n"
                f"- Practical tips\n\n"
                f"Thanks for watching! Subscribe for more."
            )

        # must have a script at this point
        if not script:
            return render_template("index.html",
                                   error="Please provide a script or a topic.",
                                   video_url=None,
                                   final_script="")

        # Split into scenes
        scenes = split_into_scenes(script)

        # Make a temp workspace for this render
        session_id = uuid.uuid4().hex
        work_dir = (IMAGES_DIR / session_id)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Generate narration
        audio_path = VIDEOS_DIR / f"{session_id}_voice.mp3"
        tts_to_mp3(script, audio_path, lang="en", speed=voice_speed)

        # Build per-scene image clips
        # Estimate per-scene durations from text length vs total narration duration
        narration = AudioFileClip(str(audio_path))
        total_audio_dur = narration.duration

        # rough weights by character count
        char_counts = [max(1, len(s)) for s in scenes]
        total_chars = sum(char_counts)
        scene_durations = [max(2.5, total_audio_dur * (c / total_chars)) for c in char_counts] # at least 2.5s per scene

        clips = []
        for scene_text, dur in zip(scenes, scene_durations):
            # choose query: if topic provided, use it; else first 5 words of scene
            query = topic if topic else " ".join(scene_text.split()[:5]) or "nature"
            raw_img = fetch_image_for_query(query, work_dir)
            img = normalize_image_to_size(raw_img, size)
            clip = make_ken_burns_clip(img, duration=dur, size=size, zoom=1.08)
            clips.append(clip)

        if not clips:
            narration.close()
            shutil.rmtree(work_dir, ignore_errors=True)
            return render_template("index.html",
                                   error="Failed to create video clips (no images).",
                                   video_url=None,
                                   final_script=script)

        video = concatenate_videoclips(clips, method="compose").set_audio(narration)

        # Optional background music
        if music_url:
            try:
                music_tmp = VIDEOS_DIR / f"{session_id}_bg.mp3"
                r = requests.get(music_url, timeout=20, stream=True)
                r.raise_for_status()
                with open(music_tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 64):
                        if chunk:
                            f.write(chunk)
                bg = AudioFileClip(str(music_tmp)).volumex(0.15)
                mixed = CompositeAudioClip([video.audio, bg])
                video = video.set_audio(mixed)
            except Exception:
                pass # ignore music errors

        # Export final video
        out_path = VIDEOS_DIR / f"{session_id}.mp4"
        video.write_videofile(
            str(out_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            verbose=False,
            logger=None
        )

        # Cleanup
        video.close()
        narration.close()
        shutil.rmtree(work_dir, ignore_errors=True)

        # Return URL to play & download + show script
        video_url = url_for('static', filename=f"videos/{out_path.name}", _external=False)
        return render_template("index.html",
                               error=None,
                               video_url=video_url,
                               final_script=script)

    # GET
    return render_template("index.html", error=None, video_url=None, final_script="")


# static file downloading (optional helper)
@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(VIDEOS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    # Local dev server
    app.run(debug=True, port=5000)

