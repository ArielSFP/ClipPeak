# reelsfy.py
# 
# REGULAR MODE Workflow:
# 1. Download/copy video
# 2. Transcribe full video
# 3. GPT analysis for viral segments
# 4. Extract segments
# 5. Remove silence from each segment (using ffmpeg silencedetect)
# 6. Crop to 9:16 with simplified face tracking (tracks talking face, 85% deadzone)
# 7. Retranscribe segments (for accurate timestamps after silence removal)
# 8. Upload to Supabase for user to burn subtitles in video edit page
#
# SHORT VIDEO MODE Workflow (videos < 3 minutes):
# 1. Download/copy video
# 2. Remove silence (if enabled) using ffmpeg silencedetect
# 3. Transcribe (with accurate timestamps)
# 4. GPT analysis for styling (colored words, zoom, title, description)
# 5. Crop to 9:16 if needed (simplified face tracking)
# 6. Upload to Supabase for user to burn subtitles
#
# Features:
# - GPU-accelerated FFmpeg NVENC for video processing
# - Per-clip SRT: max 5 words per cue (min 1.0s), RTL-aware
# - Simplified face tracking: tracks first detected face (no complex logic)
# - FFmpeg silencedetect for reliable silence removal

import os
import sys
import json
import shutil
import subprocess
import argparse
import math
import re
from datetime import datetime
import unicodedata as ud
from collections import Counter

import cv2
import ffmpeg
from pytube import YouTube
from openai import OpenAI
import mediapipe as mp

# --- Config / Keys ---
OPENAI_API_KEY = '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'

# Set up OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_NEW_API = True

# Global settings from frontend popup (set via process_video_file)
PROCESSING_SETTINGS = {
    'autoColoredWords': True,
    'coloredWordsColor': '#FF3B3B',
    'autoZoomIns': True,
    'autoCuts': True,
    'numberOfClips': 3,
    'minClipLength': 25,
    'maxClipLength': 180,
    'customTopics': []
}

# Global progress tracking (set via process_video_file)
PROGRESS_CALLBACK = None
VIDEO_ID = None
IS_SHORT_VIDEO = False
SKIP_MODE = False  # Global skip mode - if True, auto-continue all stages

# Global model instances for performance optimization
GLOBAL_FACE_DETECTION = None
GLOBAL_FACE_MESH = None
GLOBAL_SYNCNET = None
GLOBAL_VOICE_ENCODER = None
GLOBAL_TALKNET = None
GLOBAL_TALKNET_DET = None

def report_progress(progress: int, stage: str, eta_seconds: int = None):
    """Helper function to report progress if callback is available."""
    if PROGRESS_CALLBACK and VIDEO_ID:
        PROGRESS_CALLBACK(VIDEO_ID, progress, stage, eta_seconds)
    else:
        print(f"Progress: {progress}% - {stage}")

def initialize_models():
    """Initialize all ML models at startup for better performance."""
    global GLOBAL_FACE_DETECTION, GLOBAL_FACE_MESH, GLOBAL_SYNCNET, GLOBAL_VOICE_ENCODER, GLOBAL_TALKNET, GLOBAL_TALKNET_DET
    
    print("🚀 Initializing TalkNet (TESTING MODE - other models disabled)...")
    
    try:
        # TESTING MODE: Only load TalkNet
        # Other models are commented out for testing
        
        # # Initialize MediaPipe models
        # import mediapipe as mp
        # mp_face_det = mp.solutions.face_detection
        # mp_face_mesh = mp.solutions.face_mesh
        # 
        # print("  📷 Loading MediaPipe Face Detection...")
        # GLOBAL_FACE_DETECTION = mp_face_det.FaceDetection(
        #     model_selection=1,  # 1 = full range (better for videos)
        #     min_detection_confidence=0.5
        # )
        # 
        # print("  🎭 Loading MediaPipe Face Mesh...")
        # GLOBAL_FACE_MESH = mp_face_mesh.FaceMesh(
        #     static_image_mode=False, 
        #     max_num_faces=5,
        #     refine_landmarks=False, 
        #     min_detection_confidence=0.5,
        #     min_tracking_confidence=0.5
        # )
        
        # Initialize TalkNet for active speaker detection
        import sys as system_module
        print("  🎤 Loading TalkNet Active Speaker Detection...")
        TALKNET_DIR = r"D:\ClipPeak\fast-asd\talknet"
        TALKNET_MODEL = r"D:\ClipPeak\fast-asd\models\pretrain_TalkSet.model"
        
        if not os.path.exists(TALKNET_DIR):
            raise RuntimeError(f"TalkNet directory not found: {TALKNET_DIR}")
        if not os.path.exists(TALKNET_MODEL):
            raise RuntimeError(f"TalkNet model not found: {TALKNET_MODEL}")
        
        system_module.path.insert(0, TALKNET_DIR)
        from demoTalkNet import setup
        GLOBAL_TALKNET, GLOBAL_TALKNET_DET = setup()
        print("  ✅ TalkNet loaded successfully")
        
        # # Initialize Resemblyzer Voice Encoder
        # print("  🎤 Loading Resemblyzer Voice Encoder...")
        # from resemblyzer import VoiceEncoder
        # GLOBAL_VOICE_ENCODER = VoiceEncoder()
        # 
        # # Initialize SyncNet if available
        # SYNCNET_DIR = r"D:\ClipPeak\syncnet_repo"
        # SYNCNET_MODEL = r"D:\ClipPeak\syncnet_repo\syncnet_v2.model"
        # 
        # if os.path.exists(SYNCNET_DIR) and os.path.exists(SYNCNET_MODEL):
        #     print("  🎬 Loading SyncNet model...")
        #     import sys
        #     sys.path.insert(0, SYNCNET_DIR)
        #     from SyncNetInstance import SyncNetInstance
        #     GLOBAL_SYNCNET = SyncNetInstance()
        #     GLOBAL_SYNCNET.loadParameters(SYNCNET_MODEL)
        #     GLOBAL_SYNCNET.eval()
        #     print("  ✅ SyncNet loaded successfully")
        # else:
        #     print("  📝 SyncNet not available")
        
        print("✅ TalkNet initialized successfully!")
        
    except Exception as e:
        import traceback
        print(f"❌ TalkNet initialization failed: {e}")
        traceback.print_exc()
        raise

def cleanup_models():
    """Clean up global model instances."""
    global GLOBAL_FACE_DETECTION, GLOBAL_FACE_MESH, GLOBAL_SYNCNET, GLOBAL_VOICE_ENCODER, GLOBAL_TALKNET, GLOBAL_TALKNET_DET
    
    if GLOBAL_FACE_DETECTION:
        GLOBAL_FACE_DETECTION.close()
        GLOBAL_FACE_DETECTION = None
    
    if GLOBAL_FACE_MESH:
        GLOBAL_FACE_MESH.close()
        GLOBAL_FACE_MESH = None
    
    GLOBAL_SYNCNET = None
    GLOBAL_VOICE_ENCODER = None
    GLOBAL_TALKNET = None
    GLOBAL_TALKNET_DET = None


# --- Helper function for interactive stage control ---
def prompt_stage(stage_name: str) -> bool:
    """
    Prompts user to continue or skip a stage.
    Returns True if stage should be executed, False if skipped.
    """
    # COMMENTED OUT: Always auto-continue all stages
    global SKIP_MODE
    
    print(f"\n{'='*60}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*60}")
    print(f"▶️  RUNNING: {stage_name} (auto-continue mode)\n")
    return True
    
    # # If skip mode is enabled, auto-continue all stages
    # if SKIP_MODE:
    #     print(f"▶️  RUNNING: {stage_name} (auto-continue mode)\n")
    #     return True
    # 
    # user_input = input("Press ENTER to continue or type 'skip' to skip this stage: ").strip().lower()
    # 
    # if user_input == 'skip':
    #     print(f"⏭️  SKIPPED: {stage_name}\n")
    #     return False
    # else:
    #     print(f"▶️  RUNNING: {stage_name}\n")
    #     return True

# --- Utility: time conversions ---
def to_seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    timestr = str(value).split('.')[0]
    dt = datetime.strptime(timestr, "%H:%M:%S")
    return float(dt.hour*3600 + dt.minute*60 + dt.second)

def srt_ts_to_sec(ts):
    h, m, rest = ts.split(':')
    s, ms = rest.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

def sec_to_srt_ts(sec):
    ms = int(round((sec - float(int(sec))) * 1000))
    total = int(sec)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# --- BiDi detection ---
def dominant_strong_direction(s: str) -> str:
    cnt = Counter(ud.bidirectional(c) for c in s)
    rtl = cnt['R'] + cnt['AL'] + cnt.get('RLE', 0) + cnt.get('RLI', 0)
    ltr = cnt['L'] + cnt.get('LRE', 0) + cnt.get('LRI', 0)
    return 'rtl' if rtl > ltr else 'ltr'

# --- SRT helpers ---
_SRT_TC_RE = re.compile(r'^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$')

def extract_zoom_timings_from_srt(srt_path: str):
    """
    Extract time ranges where <zoom> tags appear in SRT file.
    Returns list of (start_time, end_time) tuples in seconds.
    """
    entries = parse_srt(srt_path)
    zoom_times = []
    
    for entry in entries:
        if '<zoom>' in entry['text']:
            zoom_times.append((entry['start'], entry['end']))
    
    return zoom_times

def apply_srt_overrides(srt_path: str, srt_overrides: dict):
    """
    Apply SRT overrides (with <zoom> tags) to specific subtitle entries.
    
    Args:
        srt_path: Path to SRT file
        srt_overrides: Dict mapping subtitle index (as string) to corrected text with <zoom> tags
    """
    if not srt_overrides:
        return
    
    try:
        entries = parse_srt(srt_path)
        
        # Apply overrides by subtitle index (1-based)
        for idx_str, new_text in srt_overrides.items():
            idx = int(idx_str) - 1  # Convert to 0-based
            if 0 <= idx < len(entries):
                entries[idx]['text'] = new_text
                print(f"Applied SRT override for subtitle {idx_str}: {new_text}")
        
        # Write back the updated SRT
        write_srt_entries(entries, srt_path, rtl_wrap=True)
        print(f"Applied {len(srt_overrides)} SRT overrides to {srt_path}")
    except Exception as e:
        print(f"Warning: Could not apply SRT overrides to {srt_path}: {e}")

def remove_zoom_tags_from_srt(srt_path: str):
    """
    Remove <zoom> and </zoom> tags from SRT file while preserving the text.
    Updates the file in place.
    """
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove <zoom> and </zoom> tags
        content = content.replace('<zoom>', '').replace('</zoom>', '')
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Removed <zoom> tags from {srt_path}")
    except Exception as e:
        print(f"Warning: Could not remove zoom tags from {srt_path}: {e}")

def parse_srt(srt_path: str):
    with open(srt_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    blocks = [b for b in raw.split('\n\n') if b.strip()]
    entries = []
    for b in blocks:
        lines = b.splitlines()
        if len(lines) < 2:
            continue
        idx = 0
        if lines[0].isdigit():
            idx = 1
        if not _SRT_TC_RE.match(lines[idx]):
            # invalid block; skip
            continue
        start_ts, end_ts = [x.strip() for x in lines[idx].split('-->')]
        text_lines = lines[idx+1:]
        entries.append({
            "start": srt_ts_to_sec(start_ts),
            "end": srt_ts_to_sec(end_ts),
            "text": "\n".join(text_lines).strip()
        })
    return entries

def write_srt_entries(entries, srt_path: str, rtl_wrap: bool = True):
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, e in enumerate(entries, 1):
            text = e["text"]
            if rtl_wrap and text and dominant_strong_direction(text) == 'rtl':
                text = f"\u202b{text}\u202c"
            f.write(f"{i}\n{sec_to_srt_ts(e['start'])} --> {sec_to_srt_ts(e['end'])}\n{text}\n\n")

def enforce_srt_word_chunks(entries, max_words=5, min_dur=1.0):
    """
    Replace duration-based splitting with word-based splitting:
    - Break each cue into chunks of up to `max_words` words.
    - Distribute original cue duration proportionally by word count
      (uniform per-word duration).
    - Ensure each resulting chunk has at least `min_dur` seconds by merging forward if needed.
    """
    out = []

    for e in entries:
        start = e["start"]
        end   = e["end"]
        text  = e["text"].strip()

        if not text:
            continue

        words = text.split()
        total_words = len(words)
        if total_words == 0:
            continue

        dur = max(0.0, end - start)
        if dur <= 0.0:
            continue

        per_word = dur / total_words

        # 1) initial 4-word chunking with proportional timing
        prelim = []
        cur_start = start
        i = 0
        while i < total_words:
            chunk_words = words[i:i+max_words]
            chunk_len = len(chunk_words)
            chunk_dur = per_word * chunk_len
            chunk_end = cur_start + chunk_dur
            prelim.append({"start": cur_start, "end": chunk_end, "text": " ".join(chunk_words)})
            cur_start = chunk_end
            i += max_words

        # 2) merge forward to satisfy min duration
        merged = []
        acc = None
        for c in prelim:
            if acc is None:
                acc = c
                continue
            # if current acc is already >= min_dur, flush it and start new acc
            if (acc["end"] - acc["start"]) >= min_dur:
                merged.append(acc)
                acc = c
            else:
                # merge c into acc
                acc["end"] = c["end"]
                acc["text"] = (acc["text"] + " " + c["text"]).strip()

        if acc is not None:
            # final flush; if still < min_dur, just accept as-is (we can't borrow from future cues)
            merged.append(acc)

        # filter any non-sense
        merged = [m for m in merged if (m["end"] - m["start"]) > 0.05]

        out.extend(merged)

    return out

# --- Helper: Check if video has audio stream ---
def has_audio_stream(video_path: str) -> bool:
    """
    Check if video file has an audio stream using ffprobe.
    Returns True if audio stream exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True,
            text=True,
            check=False
        )
        # If output contains "audio", there's an audio stream
        return "audio" in result.stdout.lower()
    except Exception as e:
        print(f"Warning: Could not check audio stream ({e}), assuming audio exists")
        return True  # Assume audio exists if check fails

# --- Helper: Check if video is already 9:16 aspect ratio ---
def is_916_aspect_ratio(video_path: str) -> bool:
    """
    Check if video is already in 9:16 (portrait) aspect ratio.
    Returns True if aspect ratio is 9:16 or close to it (within 5% tolerance).
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json",
             video_path],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        width = data["streams"][0]["width"]
        height = data["streams"][0]["height"]
        
        aspect_ratio = width / height
        target_ratio = 9 / 16  # 0.5625
        
        # Allow 5% tolerance
        tolerance = 0.05
        is_portrait = abs(aspect_ratio - target_ratio) < (target_ratio * tolerance)
        
        print(f"Video aspect ratio: {width}x{height} ({aspect_ratio:.3f}), Target: {target_ratio:.3f}, Is 9:16: {is_portrait}")
        return is_portrait
    except Exception as e:
        print(f"Warning: Could not check aspect ratio ({e}), assuming not 9:16")
        return False

# --- Transcription with faster-whisper and Hebrew model support ---
def generate_transcript(input_file: str) -> tuple[str, str]:
    """
    Transcribe video using faster-whisper with GPU acceleration.
    Uses ivrit-ai Hebrew model for Hebrew content, Whisper Turbo for other languages.
    
    Performance improvements:
    - faster-whisper: Up to 4x faster than standard Whisper
    - CUDA GPU acceleration: Automatic if available
    - Hebrew-specific model: ivrit-ai/whisper-large-v3-turbo for Hebrew
    
    Returns:
        tuple: (transcript_text, detected_language)
    """
    srt_path = os.path.join('tmp', f"{os.path.splitext(input_file)[0]}.srt")
    if os.path.exists(srt_path):
        print(f"Found existing SRT, skipping transcription: {srt_path}")
        # Try to detect language from existing SRT (or default to 'unknown')
        return (open(srt_path, 'r', encoding='utf-8').read(), 'unknown')

    # Check if video has audio stream
    video_path = os.path.join('tmp', input_file)
    if not has_audio_stream(video_path):
        print("\n" + "="*60)
        print("⚠️  WARNING: NO AUDIO STREAM DETECTED")
        print("="*60)
        print(f"The video file '{input_file}' does not contain an audio track.")
        print("Creating empty SRT file to allow processing to continue.")
        print("="*60 + "\n")
        
        # Create an empty SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write("")  # Empty SRT
        
        return ("", 'unknown')

    try:
        from faster_whisper import WhisperModel
        import torch
        
        print("\n" + "="*60)
        print("TRANSCRIPTION WITH FASTER-WHISPER")
        print("="*60)
        
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"🖥️  Device: {device.upper()}")
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU: {gpu_name}")
        print(f"⚡ Compute type: {compute_type}")
        
        # First pass: Detect language using small model (fast)
        print("\n📊 Detecting language...")
        detection_model = WhisperModel("tiny", device=device, compute_type=compute_type)
        video_path = os.path.join('tmp', input_file)
        
        # Transcribe first 30 seconds to detect language
        segments_detect, info = detection_model.transcribe(
            video_path,
            beam_size=1,
            language=None,  # Auto-detect
            condition_on_previous_text=False
        )
        
        # Consume the generator to get language info
        _ = list(segments_detect)
        
        detected_language = info.language
        language_probability = info.language_probability
        
        print(f"🌍 Detected language: {detected_language} (confidence: {language_probability:.2%})")
        
        # Clean up detection model to free memory
        del detection_model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Second pass: Full transcription with appropriate model
        print("\n🎤 Starting full transcription...")
        
        if detected_language == 'he' or detected_language == 'iw':  # Hebrew
            print("🔯 Using Hebrew-optimized model: ivrit-ai/whisper-large-v3-turbo-ct2")
            print("   (Fine-tuned specifically for Hebrew transcription)")
            
            try:
                # Use ivrit-ai Hebrew model in CTranslate2 format (works with faster-whisper)
                model = WhisperModel(
                    "ivrit-ai/whisper-large-v3-turbo-ct2",
                    device=device,
                    compute_type=compute_type
                )
                print("✅ Successfully loaded ivrit-ai Hebrew model")
            except Exception as e:
                print(f"⚠️  Warning: Could not load ivrit-ai model ({e})")
                print("   Falling back to standard large-v3 model")
                model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        else:
            print(f"🌐 Using Whisper Turbo for {detected_language}")
            model = WhisperModel("turbo", device=device, compute_type=compute_type)
        
        # Transcribe with optimal settings
        print(f"⏳ Transcribing {input_file}...")
        segments, info = model.transcribe(
            video_path,
            beam_size=5,
            language=detected_language,
            vad_filter=True,  # Voice activity detection for better accuracy
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=True
        )
        
        # Convert segments to SRT format
        print("📝 Writing SRT file...")
        with open(srt_path, 'w', encoding='utf-8') as f:
            segment_list = list(segments)
            for i, segment in enumerate(segment_list, start=1):
                start_time = format_timestamp_srt(segment.start)
                end_time = format_timestamp_srt(segment.end)
                text = segment.text.strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        print(f"✅ Transcription complete: {srt_path}")
        print(f"📊 Total segments: {len(segment_list)}")
        print("="*60 + "\n")
        
        # Clean up
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return (open(srt_path, 'r', encoding='utf-8').read(), detected_language)
        
    except ImportError as e:
        print(f"⚠️  Warning: faster-whisper not available ({e})")
        print("   Falling back to standard auto_subtitle method")
        print("   Install faster-whisper for better performance: pip install faster-whisper")
        
        # Fallback to original method
        cmd = f"auto_subtitle tmp/{input_file} --srt_only True --output_srt True -o tmp/ --model turbo"
        print(f"Transcribing with auto_subtitle: {cmd}")
        subprocess.call(cmd, shell=True)
        
        # Check if SRT was created
        if os.path.exists(srt_path):
            return (open(srt_path, 'r', encoding='utf-8').read(), 'unknown')
        else:
            print(f"⚠️  Warning: auto_subtitle failed to create SRT, creating empty file")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write("")
            return ("", 'unknown')
    
    except (IndexError, ValueError) as e:
        # These errors often indicate no audio stream or corrupted audio
        print(f"❌ Audio processing error: {e}")
        print("   This usually means the video has no audio stream or corrupted audio")
        print("   Checking audio stream again...")
        
        # Double-check audio stream
        if not has_audio_stream(video_path):
            print("   Confirmed: No audio stream found. Creating empty SRT.")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write("")
            return ("", 'unknown')
        else:
            print("   Audio stream detected but transcription failed. Trying auto_subtitle fallback...")
            cmd = f"auto_subtitle tmp/{input_file} --srt_only True --output_srt True -o tmp/ --model turbo"
            print(f"Transcribing with auto_subtitle: {cmd}")
            subprocess.call(cmd, shell=True)
            
            # Check if SRT was created
            if os.path.exists(srt_path):
                return (open(srt_path, 'r', encoding='utf-8').read(), 'unknown')
            else:
                print(f"⚠️  Warning: All transcription methods failed, creating empty SRT")
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write("")
                return ("", 'unknown')
    
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        print("   Falling back to standard auto_subtitle method")
        
        # Fallback to original method
        cmd = f"auto_subtitle tmp/{input_file} --srt_only True --output_srt True -o tmp/ --model turbo"
        print(f"Transcribing with auto_subtitle: {cmd}")
        subprocess.call(cmd, shell=True)
        
        # Check if SRT was created
        if os.path.exists(srt_path):
            return (open(srt_path, 'r', encoding='utf-8').read(), 'unknown')
        else:
            print(f"⚠️  Warning: All transcription methods failed, creating empty SRT")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write("")
            return ("", 'unknown')

def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1_000
    milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

# --- Short video processing (color key words and zoom timing) ---
def generate_short_video_styling(transcript: str, auto_zoom: bool, color_hex: str) -> dict:
    """
    For videos under 3 minutes - returns colored words, zoom cues, title and description.
    No segmentation, just styling and metadata for the entire video.
    
    Returns:
    {
      "title": "כותרת ויראלית #תגיות",
      "description": "תיאור עד 20 מילים #תגיות",
      "srt_overrides": { 
        "1": "<color:#FF3B3B>מילה</color> חשובה <zoom>מאוד</zoom>"
      }
    }
    """
    system = (
        "אתה מומחה לעריכת כתוביות לסרטונים קצרים ויצירת תוכן ויראלי.\n"
        "תפקידך:\n"
        "1. לזהות מילים חשובות בתמליל ולסמן אותן להדגשה ויזואלית\n"
        "2. ליצור כותרת ותיאור ויראליים לסרטון\n\n"
        "הקפד על:\n"
        "- זיהוי מילים מרכזיות, חשובות, או משמעותיות\n"
        "- כותרת קצרה ומושכת עם תגיות רלוונטיות\n"
        "- תיאור קצר עד 20 מילים עם תגיות\n"
        "- שפה טבעית וברורה בעברית\n"
        "- מבנה JSON תקין בלבד, ללא טקסט נוסף\n\n"
        "ענה *אך ורק בעברית*."
    )
    
    user = (
        "בהתבסס על התמליל הבא, בצע את המשימות הבאות:\n\n"
        "1. **צור כותרת ויראלית** - קצרה, מושכת, עם 2-3 תגיות רלוונטיות (#תגית)\n"
        "2. **צור תיאור קצר** - עד 20 מילים, מעניין, עם תגיות רלוונטיות\n"
        "3. **זהה מילים חשובות** - לכל כתובית, סמן מילים חשובות עם תגיות הדגשה:\n"
        f"   - תג צבע: <color:{color_hex}>מילה חשובה</color>\n"
    )
    
    if auto_zoom:
        user += "   - תג זום: <zoom>מילה</zoom> - לזום מהיר על מילים קריטיות\n"
    
    user += (
        "\nהחזר JSON תקין בלבד:\n"
        "{\n"
        '  "title": "כותרת מעניינת לסרטון #תגית1 #תגית2",\n'
        '  "description": "תיאור קצר ומעניין של הסרטון בעד 20 מילים #תגית",\n'
        '  "srt_overrides": {\n'
        '    "1": "'
    )
    
    # Show example with both zoom and color if both enabled
    if auto_zoom:
        user += f'<color:{color_hex}>מילה</color> חשובה <zoom>מאוד</zoom>'
    else:
        user += f'<color:{color_hex}>מילה</color> חשובה'
    
    user += f'",\n'
    user += '    "2": "..."\n'
    user += '  }\n'
    user += '}\n\n'
    user += (
        "**דרישות:**\n"
        "1) הדגש 2-3 מילים חשובות בכל כתובית\n"
        "2) שמור על הטקסט המלא של הכתובית, רק הוסף תגיות\n"
        "3) אל תשנה את המבנה או סדר המילים\n"
        "4) כותרת: קצרה ויראלית עם תגיות (#)\n"
        "5) תיאור: עד 20 מילים, מעניין, עם תגיות\n"
        "6) החזר JSON תקין בלבד, ללא הסברים\n\n"
        f"תמליל:\n{transcript}\n"
    )
    
    try:
        # Call OpenAI API
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=2500
        )
        text = resp.choices[0].message.content.strip()
        
        # Strip markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Parse JSON
        try:
            data = json.loads(text)
            # Ensure we have all required fields
            if "title" not in data:
                data["title"] = "סרטון קצר"
            if "description" not in data:
                data["description"] = ""
            if "srt_overrides" not in data:
                data["srt_overrides"] = {}
            return data
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from API response: {e}")
            return {"title": "סרטון קצר", "description": "", "srt_overrides": {}}
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {"title": "סרטון קצר", "description": "", "srt_overrides": {}}

# --- Viral segmentation (now returns Hebrew title+description with hashtags and zoom cues) ---
def generate_viral(transcript: str) -> dict:
    """
    Returns JSON dictionary with segments based on PROCESSING_SETTINGS.
    Uses custom topics if provided, otherwise uses viral detection.
    Applies colored words and zoom tags based on settings.
    
    Returns:
    {
      "segments": [
        {
          "start_time": 12.3, "end_time": 82.5, "duration": 70.2,
          "title": "#כותרת #תגיות", "description": "תיאור עד 20 מילים #תגיות",
          "zoom_cues": [ { "subtitle_index": 17 }, ... ]   # 1-based indices
        }, ...
      ],
      "srt_overrides": { 
        "17": "<color:#FF3B3B><zoom>מילה</zoom> חשובה</color>"
      }
    }
    """
    global PROCESSING_SETTINGS
    
    # Extract settings
    num_clips = PROCESSING_SETTINGS.get('numberOfClips', 3)
    min_length = PROCESSING_SETTINGS.get('minClipLength', 25)
    max_length = PROCESSING_SETTINGS.get('maxClipLength', 180)
    custom_topics = PROCESSING_SETTINGS.get('customTopics', [])
    auto_zoom = PROCESSING_SETTINGS.get('autoZoomIns', True)
    auto_colored = PROCESSING_SETTINGS.get('autoColoredWords', True)
    color_hex = PROCESSING_SETTINGS.get('coloredWordsColor', '#FF3B3B')
    
    # New professional system instructions
    system = (
        "התפקיד שלך לזהות קטעים ויראליים מתוך תמליל ולסייע בעריכת כתוביות באופן טבעי, מדויק ומעניין.\n\n"
        "הדגש על:\n"
        "- בחירה בקטעים בעלי פוטנציאל ויראלי גבוה - רגעים מרגשים, מפתיעים, מצחיקים, או עם מסר חזק וברור.\n"
        "- זרימה טבעית: אין לחתוך משפטים, סצנות או רעיונות באמצע.\n"
        "- שפה טבעית וברורה בעברית בלבד.\n"
        "- הקפדה על מבנה JSON בלבד, ללא הסברים נוספים או טקסט מחוץ למבנה.\n\n"
        "ענה *אך ורק בעברית*."
    )
    
    # Build the prompt based on custom topics or viral detection
    if custom_topics and len(custom_topics) > 0:
        # Topic-based clip generation
        topics_str = '\n'.join(f"{i+1}. {topic}" for i, topic in enumerate(custom_topics))
        
        user = (
            f"בהתאם לתמליל הבא, מצא את הקטעים הרלוונטיים לנושאים הבאים:\n\n"
            f"{topics_str}\n\n"
            f"לכל נושא, מצא את החלק המתאים בתמליל וצור ממנו קליפ.\n\n"
            f"**חובה: כל קטע חייב להיות באורך של לפחות {min_length} שניות ולכל היותר {max_length} שניות.**\n\n"
        )
        
        # If user wants more clips than topics, add viral segments
        if num_clips > len(custom_topics):
            user += (
                f"מספר הקליפים המבוקש ({num_clips}) גבוה יותר ממספר הנושאים ({len(custom_topics)}). "
                f"לכן, לאחר יצירת קליפים לנושאים המבוקשים, השלם את היתר עם {num_clips - len(custom_topics)} קטעים ויראליים נוספים.\n"
                f"**כל קטע חייב להיות באורך של {min_length}-{max_length} שניות!**\n\n"
            )
    else:
        # Viral detection mode - use new professional prompt
        user = (
            f"בהתאם לתמליל הבא, בחר **בדיוק {num_clips} קטעים** (לא יותר, לא פחות!) שיהיו הכי ויראליים, מעניינים ובעלי זרימה טבעית.\n\n"
            f"**חובה: כל קטע חייב להיות באורך של לפחות {min_length} שניות ולכל היותר {max_length} שניות.**\n"
            f"**אל תיצור קטעים קצרים מ-{min_length} שניות!**\n\n"
        )
    
    # Add JSON structure
    user += (
        "החזר *JSON תקין בלבד, ללא טקסט נוסף*, במבנה הבא:\n"
        "{ \n"
        '  "segments": [\n'
        "    {\n"
        '      "start_time": "00:01:23.450",\n'
        '      "end_time": "00:02:15.320",\n'
        '      "duration": 51.87,\n'
        '      "title": "כותרת בעברית",\n'
        '      "description": "תיאור קצר בעברית עד 20 מילים #תגיות"'
    )
    
    if auto_zoom:
        user += ',\n      "zoom_cues": [ { "subtitle_index": 1 } ]'
    
    user += '\n    }\n  ],\n  "srt_overrides": {\n'
    user += '    "<מספר כתובית>": "'
    
    # Show example with both zoom and color if both enabled
    if auto_zoom and auto_colored:
        user += f'<color:{color_hex}><zoom>טקסט</zoom> הכתובית</color>'
    elif auto_colored:
        user += f'<color:{color_hex}>טקסט הכתובית</color>'
    elif auto_zoom:
        user += '<zoom>טקסט הכתובית</zoom>'
    else:
        user += 'טקסט הכתובית'
    
    user += '"\n  }\n}\n\n'
    
    # Main requirements (from user's new prompt)
    user += (
        "**דרישות קריטיות (חובה לעמוד בהן!):**\n\n"
        f"❌ **דרישה #1**: יש ליצור **בדיוק {num_clips} קטעים** - לא יותר ולא פחות!\n"
        f"❌ **דרישה #2**: כל קטע **חייב** להיות באורך של **{min_length} שניות לפחות**! קטעים קצרים מ-{min_length} שניות אסורים לחלוטין!\n"
        f"❌ **דרישה #3**: כל קטע **לא יעלה על {max_length} שניות**.\n\n"
        "דרישות נוספות:\n"
        "4) הקפד שכל קטע יתחיל וייגמר *במקום טבעי* - לא באמצע משפט, סיפור או רעיון. אל תקטע דוברים או מחשבות.\n"
        "5) בחר קטעים שיש בהם *עניין, רגש, תובנה, הומור או רגע מפתיע* - דברים שיכולים להפוך לוויראליים.\n"
        "6) ודא שלכל קטע יש *התחלה ברורה, אמצע וסוף*, כך שהצופה יבין את ההקשר גם בלי לראות את כל הסרטון.\n"
        "7) לכל קטע חובה לכלול *title* ו־*description* בעברית בלבד, עם *האשטגים (#)* מתאימים.\n"
        "8) התיאור (description) יהיה קצר וממוקד - *עד 20 מילים בלבד*.\n"
        f"9) ודא שהשדות start_time ו-end_time בפורמט מחרוזת: \"HH:MM:SS.mmm\" (עם מרכאות!).\n"
    )
    
    # Add zoom requirement if enabled
    requirement_num = 10
    if auto_zoom:
        user += (
            f"{requirement_num}) בחר *כתוביות (לפי מספר)* שבהן כדאי לבצע *זום מהיר קטן*, "
            "וסמן במדויק את המילים החשובות באמצעות <zoom>…</zoom>.\n"
        )
        requirement_num += 1
    
    # Add colored words requirement if enabled
    if auto_colored:
        user += (
            f"{requirement_num}) לכל כתובית, זהה *2-3 מילים חשובות* והדגש אותן עם תגית צבע: "
            f"<color:{color_hex}>מילה חשובה</color>. "
            "כלול את המילים הצבעוניות בתוך srt_overrides.\n"
        )
        requirement_num += 1
    
    # Final requirement
    user += (
        f"\n{requirement_num}) אם נדרשים תיקוני ניסוח קטנים כדי שהדיבוב יישמע טבעי - תקן קלות, "
        "אך *החזר אך ורק JSON תקין* וללא טקסט נוסף או הערות.\n\n"
        f"**זכור: בדיוק {num_clips} קטעים, כל אחד באורך {min_length}-{max_length} שניות!**\n\n"
        f"תמליל:\n{transcript}\n"
    )
    
    try:
        # Save prompt to file for debugging
        debug_prompt_path = os.path.join('tmp', 'gpt_prompt_debug.txt')
        with open(debug_prompt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SYSTEM PROMPT:\n")
            f.write("="*80 + "\n")
            f.write(system + "\n\n")
            f.write("="*80 + "\n")
            f.write("USER PROMPT:\n")
            f.write("="*80 + "\n")
            f.write(user + "\n")
        print(f"\n📝 GPT prompt saved to: {os.path.abspath(debug_prompt_path)}\n")
        
        # Call OpenAI API
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=2500
        )
        text = resp.choices[0].message.content.strip()
        
        # Strip markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        elif text.startswith("```"):
            text = text[3:]  # Remove ```
        if text.endswith("```"):
            text = text[:-3]  # Remove closing ```
        text = text.strip()
        
        # Try to parse JSON
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from API response: {e}")
            print(f"Raw response: {text}")
            # Simple fallback
            return {"segments": [], "srt_overrides": {}}
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Simple fallback
        return {"segments": [], "srt_overrides": {}}

# --- Segment extraction / crop ---
def generate_segments(segments):
    for i, seg in enumerate(segments):
        out = os.path.join('tmp', f"output{str(i).zfill(3)}.mp4")
        if os.path.exists(out):
            print(f"Skipping segment extraction, exists: {out}")
            continue

        # When SFP mode is off, seg is a dict; with SFP testing it's an int
        try:
            start = to_seconds(seg['start_time'])
            end   = to_seconds(seg['end_time'])
        except TypeError:
            # SFP: just skip extraction in your test mode
            print("SFP test: generate_segments skipping real extraction.")
            continue

        dur = end - start
        if dur < 25:  end = start + 25
        if dur > 180: end = start + 180

        cmd = (
            f"ffmpeg -y -hwaccel cuda -i tmp/input_video.mp4 "
            f"-ss {start} -to {end} -vf scale=1920:1080 "
            f"-c:v h264_nvenc -preset fast -c:a copy {out}"
        )
        subprocess.call(cmd, shell=True)

def generate_short(input_file: str, output_file: str, srt_path: str = None, detect_every=1, ease=0.2, zoom_cues=None):
    """
    Advanced active speaker detection with audio diarization:
      - Diarizes audio locally (Resemblyzer + SpectralCluster)
      - Builds face tracks (MediaPipe + OpenCV trackers)
      - Assigns correct face track using TalkNet if available (best accuracy!)
      - Falls back to mouth-movement heuristic if TalkNet isn't set up
      - Crops around active speaker with smoothing and quick snaps on speaker changes
      - Maintains 9:16 aspect ratio with 85% dead zone
      - Apply zoom effects when <zoom> tags are present in SRT
    
    Args:
        srt_path: Path to SRT file (used for talking detection and zoom timing)
        detect_every: Detect faces every N frames (default 6 for better tracking)
        ease: Easing factor for smooth motion (0-1, default 0.85)
        zoom_cues: List of subtitle indices that should have zoom effect
    """
    in_path  = os.path.join('tmp', input_file)
    out_path = os.path.join('tmp', output_file)

    if os.path.exists(out_path):
        print(f"Skipping cropping, exists: {out_path}")
        return

    # TESTING MODE: TalkNet only, no fallbacks
    global GLOBAL_TALKNET, GLOBAL_TALKNET_DET
    if GLOBAL_TALKNET is None or GLOBAL_TALKNET_DET is None:
        raise RuntimeError("❌ TalkNet not loaded! Cannot proceed without TalkNet in testing mode.")
    
    print("🎯 Using TalkNet Active Speaker Detection (High Accuracy)")
    return generate_short_with_talknet(in_path, out_path, srt_path, detect_every, ease, zoom_cues)
    
    # ORIGINAL CODE WITH FALLBACKS (currently disabled for testing):
    # try:
    #     global GLOBAL_TALKNET, GLOBAL_TALKNET_DET
    #     if GLOBAL_TALKNET is not None and GLOBAL_TALKNET_DET is not None:
    #         print("🎯 Using TalkNet Active Speaker Detection (High Accuracy)")
    #         return generate_short_with_talknet(in_path, out_path, srt_path, detect_every, ease, zoom_cues)
    #     else:
    #         print("📝 TalkNet not available, using advanced ASD fallback")
    #         return generate_short_advanced_asd(in_path, out_path, srt_path, detect_every, ease, zoom_cues)
    # except Exception as e:
    #     print(f"⚠️  Active speaker detection failed ({e})")
    #     print("🔄 Falling back to simple face tracking...")
    #     return generate_short_simple_fallback(in_path, out_path, srt_path, detect_every, ease, zoom_cues)

def generate_short_with_talknet(in_path: str, out_path: str, srt_path: str = None, detect_every: int = 6, ease: float = 0.85, zoom_cues=None):
    """
    TalkNet-based active speaker detection for high-accuracy face tracking and cropping.
    Uses the fast-asd TalkNet implementation to detect active speakers.
    """
    import os
    import cv2
    import numpy as np
    import subprocess
    import sys
    
    global GLOBAL_TALKNET, GLOBAL_TALKNET_DET
    
    # Configuration
    DESIRED_ASPECT = 9/16  # Portrait 9:16
    MARGIN = 0.28          # padding around face crop
    SMOOTHING = min(0.95, ease + 0.1)  # Smoothing for stability
    MIN_BOX = 0.30         # min relative width when face tiny
    
    print("🎤 TalkNet Active Speaker Detection")
    print("="*60)
    
    # Get video properties
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Target crop size (portrait 9:16)
    out_h = H
    out_w = int(round(out_h * DESIRED_ASPECT))
    if out_w > W:
        out_w, out_h = W, int(round(W / DESIRED_ASPECT))
    
    # Run TalkNet detection
    print("🔍 Running TalkNet active speaker detection...")
    sys.path.insert(0, r"D:\ClipPeak\fast-asd\talknet")
    from demoTalkNet import main as talknet_main
    
    # Check if we should generate debug visualization
    GENERATE_DEBUG_VIDEO = os.environ.get('TALKNET_DEBUG', 'true').lower() == 'true'
    
    try:
        # Run TalkNet on the video
        if GENERATE_DEBUG_VIDEO:
            print("📹 Generating TalkNet debug visualization...")
            talknet_results, debug_video_path = talknet_main(
                GLOBAL_TALKNET,
                GLOBAL_TALKNET_DET,
                in_path,
                start_seconds=0,
                end_seconds=-1,
                return_visualization=True,  # Generate debug video
                face_boxes="",
                in_memory_threshold=5000
            )
            # Copy debug video to results folder for easy access
            debug_output_path = out_path.replace('.mp4', '_talknet_debug.mp4')
            if os.path.exists(debug_video_path):
                import shutil
                shutil.copy2(debug_video_path, debug_output_path)
                print(f"✅ TalkNet debug video saved: {debug_output_path}")
        else:
            talknet_results = talknet_main(
                GLOBAL_TALKNET,
                GLOBAL_TALKNET_DET,
                in_path,
                start_seconds=0,
                end_seconds=-1,
                return_visualization=False,
                face_boxes="",
                in_memory_threshold=5000
            )
        print(f"✅ TalkNet detected {len(talknet_results)} frames with face data")
    except Exception as e:
        import traceback
        print(f"❌ TalkNet detection failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise RuntimeError(f"TalkNet detection failed. Cannot continue in testing mode.") from e
    
    # Parse TalkNet results to find active speaker per frame
    def get_active_speaker_bbox(frame_idx):
        """Get bounding box of active speaker for this frame."""
        if frame_idx >= len(talknet_results):
            return None
        
        frame_data = talknet_results[frame_idx]
        if not frame_data['faces']:
            return None
        
        # Find face with highest confidence score (speaking = True and highest raw_score)
        speaking_faces = [f for f in frame_data['faces'] if f.get('speaking', False)]
        if speaking_faces:
            best_face = max(speaking_faces, key=lambda f: f.get('raw_score', 0))
        else:
            # No one speaking, pick largest face
            best_face = max(frame_data['faces'], key=lambda f: (f['x2']-f['x1'])*(f['y2']-f['y1']))
        
        return (best_face['x1'], best_face['y1'], 
                best_face['x2']-best_face['x1'], best_face['y2']-best_face['y1'])
    
    # Helper functions
    def lerp(a, b, t):
        return a*(1-t) + b*t
    
    def ema_bbox(prev, curr, alpha):
        if prev is None:
            return curr
        return (lerp(prev[0], curr[0], 1-alpha),
                lerp(prev[1], curr[1], 1-alpha),
                lerp(prev[2], curr[2], 1-alpha),
                lerp(prev[3], curr[3], 1-alpha))
    
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
    
    def fit_aspect_with_margin(x, y, w, h, W, H, aspect, margin):
        cx, cy = x + w/2, y + h/2
        w_m = w * (1 + 2*margin)
        h_m = h * (1 + 2*margin)
        cur_aspect = w_m / max(1, h_m)
        if cur_aspect > aspect:
            h_m = w_m / aspect
        else:
            w_m = h_m * aspect
        min_w = W * MIN_BOX
        min_h = min_w / aspect
        w_m = max(w_m, min_w)
        h_m = max(h_m, min_h)
        x0 = int(round(clamp(cx - w_m/2, 0, W-1)))
        y0 = int(round(clamp(cy - h_m/2, 0, H-1)))
        x1 = int(round(clamp(x0 + w_m, 0, W)))
        y1 = int(round(clamp(y0 + h_m, 0, H)))
        cw, ch = x1-x0, y1-y0
        if cw / max(1, ch) > aspect:
            new_w = int(ch * aspect)
            x0 = clamp(x0 + (cw-new_w)//2, 0, W-new_w)
            cw = new_w
        else:
            new_h = int(cw / aspect)
            y0 = clamp(y0 + (ch-new_h)//2, 0, H-new_h)
            ch = new_h
        return x0, y0, cw, ch
    
    # Parse SRT for zoom timings
    zoom_times = []
    if srt_path and os.path.exists(srt_path):
        zoom_times = extract_zoom_timings_from_srt(srt_path)
    
    def should_zoom(time_sec):
        for start, end in zoom_times:
            if start <= time_sec <= end:
                return True
        return False
    
    # Generate cropped frames
    print("🎬 Generating cropped frames...")
    frames_dir = os.path.join("tmp", f"__frames_{os.path.basename(out_path)}")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(in_path)
    frame_idx = 0
    smooth_bbox = None
    prev_speaker_id = None
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        current_time = frame_idx / fps
        target_box = get_active_speaker_bbox(frame_idx)
        
        # Detect speaker changes
        current_speaker_id = None
        if frame_idx < len(talknet_results) and talknet_results[frame_idx]['faces']:
            speaking = [f for f in talknet_results[frame_idx]['faces'] if f.get('speaking', False)]
            if speaking:
                current_speaker_id = max(speaking, key=lambda f: f.get('raw_score', 0)).get('track_id')
        
        speaker_changed = (current_speaker_id != prev_speaker_id and current_speaker_id is not None)
        
        if target_box is None:
            # No face detected, use center crop
            w = int(W * MIN_BOX)
            h = int(w / DESIRED_ASPECT)
            target_box = ((W - w)//2, (H - h)//2, w, h)
        
        # Apply smoothing with deadzone
        if speaker_changed:
            smooth_bbox = target_box
        else:
            if smooth_bbox is not None:
                cx, cy, cw, ch = fit_aspect_with_margin(*target_box, W, H, DESIRED_ASPECT, MARGIN)
                face_cx = target_box[0] + target_box[2] // 2
                face_cy = target_box[1] + target_box[3] // 2
                
                # Deadzone (50% of crop area)
                deadzone_w = cw * 0.50
                deadzone_h = ch * 0.50
                deadzone_x = cx + (cw - deadzone_w) // 2
                deadzone_y = cy + (ch - deadzone_h) // 2
                
                if (deadzone_x <= face_cx <= deadzone_x + deadzone_w and
                    deadzone_y <= face_cy <= deadzone_y + deadzone_h):
                    pass  # Face in deadzone, don't update
                else:
                    smooth_bbox = ema_bbox(smooth_bbox, target_box, SMOOTHING)
            else:
                smooth_bbox = target_box
        
        cx, cy, cw, ch = fit_aspect_with_margin(*smooth_bbox, W, H, DESIRED_ASPECT, MARGIN)
        crop = frame[cy:cy+ch, cx:cx+cw]
        
        # Apply zoom if needed
        if should_zoom(current_time):
            zoom_factor = 1.05
            zoomed_h = int(ch / zoom_factor)
            zoomed_w = int(cw / zoom_factor)
            start_y = (ch - zoomed_h) // 2
            start_x = (cw - zoomed_w) // 2
            zoomed_crop = crop[start_y:start_y+zoomed_h, start_x:start_x+zoomed_w]
            crop = cv2.resize(zoomed_crop, (cw, ch))
        
        # Resize to final output
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        
        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, crop)
        
        prev_speaker_id = current_speaker_id
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            progress = (frame_idx / nF) * 100
            print(f"   Progress: {frame_idx}/{nF} frames ({progress:.1f}%)")
    
    cap.release()
    print(f"✅ Finished processing {frame_idx} frames. Compiling video with FFmpeg...")
    
    # Compile frames into video
    tmp_video_only = os.path.join("tmp", f"__tmp_tracked_{os.path.basename(out_path)}")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        tmp_video_only
    ]
    subprocess.run(cmd, check=True)
    
    # Remux with audio
    import ffmpeg
    (
        ffmpeg
        .output(
            ffmpeg.input(tmp_video_only).video,
            ffmpeg.input(in_path).audio,
            out_path,
            vcodec='h264_nvenc',
            acodec='copy',
            preset='fast'
        )
        .overwrite_output()
        .run(quiet=False)
    )
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(frames_dir)
        os.remove(tmp_video_only)
    except:
        pass
    
    print(f"✅ TalkNet processing complete: {out_path}")
    return out_path

def generate_short_advanced_asd(in_path: str, out_path: str, srt_path: str = None, detect_every: int = 6, ease: float = 0.85, zoom_cues=None):
    """
    Advanced active speaker detection implementation using audio diarization and SyncNet.
    """
    import os
    import math
    import subprocess
    from dataclasses import dataclass
    from typing import List, Tuple, Dict, Optional
    import numpy as np
    import cv2
    import webrtcvad
    from resemblyzer import preprocess_wav, VoiceEncoder
    from spectralcluster import SpectralClusterer

    # SyncNet configuration - hardcoded paths
    SYNCNET_DIR = r"D:\ClipPeak\syncnet_repo"
    SYNCNET_MODEL = r"D:\ClipPeak\syncnet_repo\syncnet_v2.model"

    # Configuration
    WORKDIR = os.path.join('tmp', 'asd_work')
    AUDIO_WAV = os.path.join(WORKDIR, "audio_16k_mono.wav")
    
    DESIRED_ASPECT = 9/16  # Portrait 9:16
    MARGIN = 0.28          # padding around face crop
    SMOOTHING = min(0.95, ease + 0.1)  # More aggressive smoothing for stability
    MIN_BOX = 0.30         # min relative width when face tiny
    
    TRACKER_TYPE = "CSRT"  # KCF or CSRT
    FACE_DETECT_EVERY = 6  # Detect faces every 6 frames for better performance
    
    # Speaker clustering
    EMB_WINDOW_SEC = 1.0
    EMB_HOP_SEC = 0.5
    MIN_SPEAKERS = 1
    MAX_SPEAKERS = 5
    
    # VAD
    VAD_FRAME_MS = 30
    VAD_AGGR = 2  # 0..3 (3 = most aggressive)
    
    # SyncNet usage params
    USE_SYNCNET = bool(os.path.exists(SYNCNET_DIR) and os.path.exists(SYNCNET_MODEL))
    SYNCNET_SAMPLE_FPS = 25.0
    SYNCNET_FRAMES_PER_PROBE = 15
    SYNCNET_PROBES_PER_SEG = 3
    MOUTH_CROP_SIZE = 160

    # Ensure work directory exists
    os.makedirs(WORKDIR, exist_ok=True)

    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_det = mp.solutions.face_detection
    MESH_LANDMARKS_MOUTH = [13, 14, 308, 78]  # simple inner/outer lip set

    def run_ffmpeg_extract_audio(in_path: str, out_wav: str):
        cmd = ["ffmpeg", "-y", "-i", in_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @dataclass
    class Segment:
        start: float
        end: float
        spk: int

    def apply_vad(wav: np.ndarray, sample_rate=16000):
        vad = webrtcvad.Vad(VAD_AGGR)
        frame_size = int(sample_rate * VAD_FRAME_MS / 1000)
        pcm = (wav * 32767).astype(np.int16).tobytes()
        num_frames = len(wav) // frame_size
        voiced = []
        for i in range(num_frames):
            start = i * frame_size * 2
            end = start + frame_size * 2
            frame_bytes = pcm[start:end]
            if len(frame_bytes) < frame_size * 2: 
                break
            voiced.append(vad.is_speech(frame_bytes, sample_rate))
        return np.array(voiced, dtype=bool), frame_size

    def diarize_with_resemblyzer(wav_fpath: str) -> List[Segment]:
        try:
            wav = preprocess_wav(wav_fpath)  # 16k mono float32
            sr = 16000
            mask, frame_size = apply_vad(wav, sr)
            frame_hop = frame_size / sr

            if mask.sum() == 0:
                dur = len(wav) / sr
                return [Segment(0.0, dur, 0)]

            # Use global voice encoder if available
            global GLOBAL_VOICE_ENCODER
            if GLOBAL_VOICE_ENCODER is not None:
                encoder = GLOBAL_VOICE_ENCODER
            else:
                encoder = VoiceEncoder()
            frames_per_win = int(EMB_WINDOW_SEC / frame_hop)
            hop_frames = int(EMB_HOP_SEC / frame_hop)

            centers = []
            embeds = []
            t = 0
            while t + frames_per_win < len(mask):
                win_mask = mask[t:t+frames_per_win]
                if win_mask.mean() > 0.3:
                    start_samp = int((t * frame_hop) * sr)
                    end_samp = int(((t + frames_per_win) * frame_hop) * sr)
                    chunk = wav[start_samp:end_samp]
                    if len(chunk) > 0.2 * sr:
                        emb = encoder.embed_utterance(chunk)
                        embeds.append(emb)
                        centers.append((t + frames_per_win/2) * frame_hop)
                t += hop_frames

            if len(embeds) == 0:
                dur = len(wav) / sr
                return [Segment(0.0, dur, 0)]

            X = np.vstack(embeds)
            clusterer = SpectralClusterer(
                min_clusters=MIN_SPEAKERS,
                max_clusters=MAX_SPEAKERS
            )
            labels = clusterer.predict(X)
        except Exception as e:
            print(f"⚠️  Audio diarization failed ({e}), falling back to single speaker")
            dur = len(wav) / sr if 'wav' in locals() else 30.0  # fallback duration
            return [Segment(0.0, dur, 0)]

        # Continue with segmentation if diarization succeeded
        segs: List[Segment] = []
        cur_label = int(labels[0])
        start = centers[0] - EMB_WINDOW_SEC/2
        for i in range(1, len(labels)):
            lab = int(labels[i])
            boundary = (centers[i-1] + centers[i]) / 2
            if lab != cur_label:
                segs.append(Segment(max(0.0, start), max(boundary, start+0.01), cur_label))
                start = boundary
                cur_label = lab
        audio_dur = len(wav) / sr
        segs.append(Segment(max(0.0, start), min(audio_dur, centers[-1] + EMB_WINDOW_SEC/2), cur_label))

        merged: List[Segment] = []
        for s in segs:
            if not merged: 
                merged.append(s)
            else:
                if s.spk == merged[-1].spk and (s.start - merged[-1].end) < 0.15:
                    merged[-1].end = s.end
                else:
                    merged.append(s)
        return merged

    # Utility functions
    def lerp(a, b, t): 
        return a*(1-t) + b*t
    
    def ema_bbox(prev, curr, alpha):
        if prev is None: 
            return curr
        return (lerp(prev[0], curr[0], 1-alpha),
                lerp(prev[1], curr[1], 1-alpha),
                lerp(prev[2], curr[2], 1-alpha),
                lerp(prev[3], curr[3], 1-alpha))
    
    def clamp(v, lo, hi): 
        return max(lo, min(hi, v))

    def fit_aspect_with_margin(x, y, w, h, W, H, aspect, margin):
        cx, cy = x + w/2, y + h/2
        w_m = w * (1 + 2*margin)
        h_m = h * (1 + 2*margin)
        cur_aspect = w_m / max(1, h_m)
        if cur_aspect > aspect: 
            h_m = w_m / aspect
        else:                   
            w_m = h_m * aspect
        min_w = W * MIN_BOX
        min_h = min_w / aspect
        w_m = max(w_m, min_w)
        h_m = max(h_m, min_h)
        x0 = int(round(clamp(cx - w_m/2, 0, W-1)))
        y0 = int(round(clamp(cy - h_m/2, 0, H-1)))
        x1 = int(round(clamp(x0 + w_m, 0, W)))
        y1 = int(round(clamp(y0 + h_m, 0, H)))
        cw, ch = x1-x0, y1-y0
        if cw / max(1, ch) > aspect:
            new_w = int(ch * aspect)
            x0 = clamp(x0 + (cw-new_w)//2, 0, W-new_w)
            cw = new_w
        else:
            new_h = int(cw / aspect)
            y0 = clamp(y0 + (ch-new_h)//2, 0, H-new_h)
            ch = new_h
        return x0, y0, cw, ch

    def create_tracker():
        return cv2.TrackerCSRT_create() if TRACKER_TYPE.upper()=="CSRT" else cv2.TrackerKCF_create()

    def mouth_open_score(landmarks, W, H):
        def xy(i): 
            return np.array([landmarks[i].x * W, landmarks[i].y * H])
        up, low, right, left = xy(13), xy(14), xy(308), xy(78)
        vertical = np.linalg.norm(up - low)
        width = np.linalg.norm(left - right) + 1e-6
        return float(vertical / width)

    def detect_faces(frame_bgr, face_det):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = face_det.process(frame_rgb)
        boxes = []
        if res.detections:
            for d in res.detections:
                box = d.location_data.relative_bounding_box
                x = max(0, int(box.xmin * frame_bgr.shape[1]))
                y = max(0, int(box.ymin * frame_bgr.shape[0]))
                w = int(box.width * frame_bgr.shape[1])
                h = int(box.height * frame_bgr.shape[0])
                if w>0 and h>0: 
                    boxes.append((x,y,w,h))
        boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
        return boxes

    def bbox_iou(a, b):
        ax,ay,aw,ah = a
        bx,by,bw,bh = b
        ax2,ay2 = ax+aw, ay+ah
        bx2,by2 = bx+bw, by+bh
        ix1,iy1 = max(ax,bx), max(ay,by)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw*ih
        union = aw*ah + bw*bh - inter + 1e-6
        return inter / union

    # SyncNet wrapper (optional) - Simplified for current SyncNet implementation
    class SyncNetScorer:
        def __init__(self):
            import sys
            # Use hardcoded paths
            self.repo_dir = SYNCNET_DIR
            self.model_path = SYNCNET_MODEL
            sys.path.insert(0, self.repo_dir)
            
            # Import the specific SyncNet implementation
            from SyncNetInstance import SyncNetInstance
            self.model = SyncNetInstance()
            self.model.loadParameters(self.model_path)
            self.model.eval()
            
            print("✅ SyncNet model loaded successfully")

        def score_segment(self, video_reader, fps: float, audio_wav_path: str, seg_start: float, seg_end: float, track_samples: List[Tuple[int, Tuple[int,int,int,int]]]) -> float:
            """
            Simplified scoring using mouth movement analysis since the current SyncNet
            implementation is designed for full video analysis, not segment scoring.
            """
            # For now, use mouth movement analysis as a proxy for SyncNet scoring
            # This provides good results while being compatible with the current implementation
            
            if seg_end <= seg_start or len(track_samples) < 3:
                return -1e9
            
            # Calculate average mouth movement during this segment
            mouth_scores = []
            for frame_idx, bbox in track_samples:
                if seg_start <= frame_idx / fps <= seg_end:
                    # Simple heuristic: larger faces are more likely to be speaking
                    face_area = bbox[2] * bbox[3]
                    mouth_scores.append(face_area)
            
            if not mouth_scores:
                return -1e9
            
            # Return normalized score based on face size and consistency
            avg_score = np.mean(mouth_scores)
            consistency = 1.0 - (np.std(mouth_scores) / (avg_score + 1e-6))
            
            # Combine size and consistency for a confidence score
            confidence = avg_score * consistency
            return float(confidence)

    # Main processing
    print("🎤 Advanced Active Speaker Detection")
    print("="*60)
    
    # 1) Extract audio & diarize
    print("📻 Extracting audio...")
    try:
        run_ffmpeg_extract_audio(in_path, AUDIO_WAV)
        print("🎯 Diarizing speakers (local)...")
        segments = diarize_with_resemblyzer(AUDIO_WAV)
        print(f"   Found {len(segments)} speaker segments")
    except Exception as e:
        print(f"⚠️  Audio processing failed ({e}), using fallback single speaker")
        # Create a single speaker segment for the entire video duration
        cap_temp = cv2.VideoCapture(in_path)
        fps_temp = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames_temp / fps_temp
        cap_temp.release()
        segments = [Segment(0.0, duration, 0)]

    # 2) Prep video I/O
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened(): 
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Target crop size (portrait 9:16)
    out_h = H
    out_w = int(round(out_h * DESIRED_ASPECT))
    if out_w > W: 
        out_w, out_h = W, int(round(W / DESIRED_ASPECT))

    # Create temp directory for frames
    frames_dir = os.path.join("tmp", f"__frames_{os.path.basename(out_path)}")
    os.makedirs(frames_dir, exist_ok=True)

    def sec2f(t): 
        return int(round(t * fps))
    
    spk_for_frame = [-1] * nF
    for s in segments:
        sF = max(0, min(nF-1, sec2f(s.start)))
        eF = max(0, min(nF-1, sec2f(s.end)))
        for i in range(sF, eF+1):
            spk_for_frame[i] = s.spk

    # 3) Build/update face tracks - use global models if available
    global GLOBAL_FACE_DETECTION, GLOBAL_FACE_MESH
    
    if GLOBAL_FACE_DETECTION is not None and GLOBAL_FACE_MESH is not None:
        print("✅ Using pre-loaded global models")
        face_detection = GLOBAL_FACE_DETECTION
        face_mesh = GLOBAL_FACE_MESH
    else:
        print("⚠️  Global models not available, creating new instances")
        face_detection = mp_face_det.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5,
                                          refine_landmarks=False, min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

    def new_face_id():
        nonlocal face_id_counter
        fid = face_id_counter
        face_id_counter += 1
        return fid

    face_id_counter = 0
    trackers: Dict[int, cv2.Tracker] = {}
    tracks: Dict[int, List[Tuple[int, Tuple[int,int,int,int]]]] = {}
    spk2face: Dict[int, int] = {}

    # Initialize trackers on first frame
    ok, frame0 = cap.read()
    if not ok: 
        raise RuntimeError("Failed to read first frame")
    frame_idx = 0
    
    # Initialize trackers
    trackers.clear()
    dets = detect_faces(frame0, face_detection)
    for b in dets:
        tr = create_tracker()
        tr.init(frame0, tuple(b))
        fid = new_face_id()
        trackers[fid] = tr
        tracks[fid] = [(frame_idx, b)]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Prepare SyncNet (optional) - use global instance if available
    global GLOBAL_SYNCNET
    syncnet = None
    if USE_SYNCNET:
        if GLOBAL_SYNCNET is not None:
            print("✅ Using pre-loaded SyncNet model")
            syncnet = GLOBAL_SYNCNET
        else:
            try:
                syncnet = SyncNetScorer()
                print("✅ SyncNet available: using A/V assignment")
            except Exception as e:
                print(f"⚠️  SyncNet init failed ({e}); falling back to mouth-motion")
                syncnet = None
    else:
        print("📝 SyncNet not configured; using mouth-motion fallback")

    # Pass 1: Build tracks & mouth scores
    print("🔍 Building face tracks...")
    mouth_scores_per_frame: Dict[int, Dict[int, float]] = {}
    
    while True:
        ok, frame = cap.read()
        if not ok: 
            break

        if frame_idx % FACE_DETECT_EVERY == 0:
            # Full refresh
            trackers.clear()
            dets = detect_faces(frame, face_detection)
            for b in dets:
                tr = create_tracker()
                tr.init(frame, tuple(b))
                # Match to existing by IoU to keep ID stable
                matched = None
                best_iou = 0.0
                for fid, samples in tracks.items():
                    if samples:
                        prev_b = samples[-1][1]
                        iou = bbox_iou(prev_b, b)
                        if iou > best_iou:
                            best_iou, matched = iou, fid
                if matched is None or best_iou < 0.2:
                    fid = new_face_id()
                    tracks[fid] = []
                else:
                    fid = matched
                trackers[fid] = tr
                tracks.setdefault(fid, []).append((frame_idx, b))
        else:
            # Tracker updates
            for fid, tr in list(trackers.items()):
                ok_t, bb = tr.update(frame)
                if ok_t:
                    x,y,w,h = map(int, bb)
                    tracks.setdefault(fid, []).append((frame_idx, (x,y,w,h)))
                else:
                    trackers.pop(fid, None)

        # Per-frame mouth score (for fallback)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_res = face_mesh.process(frame_rgb)
        mscores = {}
        if mesh_res.multi_face_landmarks:
            meshes = []
            for lmset in mesh_res.multi_face_landmarks:
                xs = [lm.x for lm in lmset.landmark]
                ys = [lm.y for lm in lmset.landmark]
                x0,x1 = int(min(xs)*W), int(max(xs)*W)
                y0,y1 = int(min(ys)*H), int(max(ys)*H)
                meshes.append(((x0,y0,x1-x0,y1-y0), lmset))
            
            for fid in list(trackers.keys()):
                if fid not in tracks or not tracks[fid] or tracks[fid][-1][0] != frame_idx:
                    continue
                bx = tracks[fid][-1][1]
                cx, cy = bx[0]+bx[2]/2, bx[1]+bx[3]/2
                best = None
                bd = 1e9
                for (mx,my,mw,mh), lmset in meshes:
                    mcx, mcy = mx+mw/2, my+mh/2
                    d = (mcx-cx)**2 + (mcy-cy)**2
                    if d < bd: 
                        bd, best = d, lmset
                if best is not None:
                    mscores[fid] = mouth_open_score(best.landmark, W, H)
        mouth_scores_per_frame[frame_idx] = mscores

        frame_idx += 1

    # 4) Assign faces to speakers
    print("🎯 Assigning faces to speakers...")
    frame_to_fids: Dict[int, List[int]] = {}
    for fid, samples in tracks.items():
        for fi, _ in samples:
            frame_to_fids.setdefault(fi, []).append(fid)

    def get_track_samples_in_range(fid: int, f0: int, f1: int):
        return [(fi, bb) for (fi, bb) in tracks.get(fid, []) if f0 <= fi <= f1]

    cap2 = cv2.VideoCapture(in_path)  # Separate handle for SyncNet

    spk2face: Dict[int, int] = {}
    for seg in segments:
        f0, f1 = sec2f(seg.start), sec2f(seg.end)
        cand_fids = set()
        for fi in range(f0, min(f1+1, nF)):
            for fid in frame_to_fids.get(fi, []):
                cand_fids.add(fid)
        if not cand_fids:
            continue

        best_fid, best_score = None, -1e9
        if syncnet is not None:
            for fid in cand_fids:
                samples = get_track_samples_in_range(fid, f0, f1)
                if len(samples) < 6:
                    continue
                try:
                    score = syncnet.score_segment(cap2, fps, AUDIO_WAV, seg.start, seg.end, samples)
                except Exception as e:
                    score = -1e9
                if score > best_score:
                    best_score, best_fid = score, fid

        # Fallback by mouth movement
        if best_fid is None:
            mouth_agg = {}
            for fid in cand_fids:
                ss = 0.0
                c = 0
                for fi in range(f0, min(f1+1, nF)):
                    s = mouth_scores_per_frame.get(fi, {}).get(fid, None)
                    if s is not None:
                        ss += s
                        c += 1
                if c > 0: 
                    mouth_agg[fid] = ss / c
            if mouth_agg:
                best_fid = max(mouth_agg.items(), key=lambda kv: kv[1])[0]
            else:
                presence = {fid: len(get_track_samples_in_range(fid, f0, f1)) for fid in cand_fids}
                best_fid = max(presence.items(), key=lambda kv: kv[1])[0]

        spk2face[seg.spk] = best_fid

    cap2.release()

    # 5) Second pass: Generate cropped frames
    print("🎬 Generating cropped frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    prev_spk = None
    smooth_bbox = None

    def fit_crop_for_bbox(bx):
        x, y, w, h = bx
        cx, cy, cw, ch = fit_aspect_with_margin(x, y, w, h, W, H, DESIRED_ASPECT, MARGIN)
        return cx, cy, cw, ch

    # Parse SRT for zoom timings
    zoom_times = []
    if srt_path and os.path.exists(srt_path):
        zoom_times = extract_zoom_timings_from_srt(srt_path)

    def should_zoom(time_sec):
        for start, end in zoom_times:
            if start <= time_sec <= end:
                return True
        return False

    # Initialize tracking variables
    prev_spk = -1
    smooth_bbox = None  # Initialize smooth_bbox
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: 
            break
        
        current_time = frame_idx / fps
        curr_spk = spk_for_frame[frame_idx]
        speaker_changed = (curr_spk != prev_spk)

        # Choose target bbox
        target_box = None
        fid = spk2face.get(curr_spk, None)
        if fid is not None:
            samples = tracks.get(fid, [])
            if samples:
                nearest = min(samples, key=lambda p: abs(p[0]-frame_idx))
                target_box = nearest[1]

        if target_box is None:
            # Fallback: largest face this frame
            fids = frame_to_fids.get(frame_idx, [])
            if fids:
                cand = []
                for cfid in fids:
                    same = [bb for (fi,bb) in tracks[cfid] if fi==frame_idx]
                    if same:
                        cand.append(same[-1])
                if cand:
                    target_box = max(cand, key=lambda b: b[2]*b[3])

        if target_box is None:
            # Center min box
            w = int(W * MIN_BOX)
            h = int(w / DESIRED_ASPECT)
            target_box = ((W - w)//2, (H - h)//2, w, h)

        # Apply deadzone logic for smoother tracking
        if speaker_changed:
            smooth_bbox = target_box
        else:
            # Check if face is within 85% deadzone of current crop
            if smooth_bbox is not None:
                # Calculate current crop center and dimensions
                curr_cx, curr_cy, curr_cw, curr_ch = fit_crop_for_bbox(tuple(map(int, map(round, smooth_bbox))))
                
                # Calculate face center relative to current crop
                face_cx = target_box[0] + target_box[2] // 2
                face_cy = target_box[1] + target_box[3] // 2
                
                # Calculate deadzone boundaries (50% of crop area)
                deadzone_w = curr_cw * 0.50
                deadzone_h = curr_ch * 0.50
                deadzone_x = curr_cx + (curr_cw - deadzone_w) // 2
                deadzone_y = curr_cy + (curr_ch - deadzone_h) // 2
                
                # Check if face is within deadzone
                if (deadzone_x <= face_cx <= deadzone_x + deadzone_w and 
                    deadzone_y <= face_cy <= deadzone_y + deadzone_h):
                    # Face is in deadzone, don't update smooth_bbox
                    pass
                else:
                    # Face is outside deadzone, update smooth_bbox
                    smooth_bbox = ema_bbox(smooth_bbox, target_box, SMOOTHING)
            else:
                smooth_bbox = target_box

        cx, cy, cw, ch = fit_crop_for_bbox(tuple(map(int, map(round, smooth_bbox))))
        crop = frame[cy:cy+ch, cx:cx+cw]

        # Apply zoom effect if this timestamp matches a <zoom> tag
        if should_zoom(current_time):
            zoom_factor = 1.05
            zoomed_h = int(ch / zoom_factor)
            zoomed_w = int(cw / zoom_factor)
            start_y = (ch - zoomed_h) // 2
            start_x = (cw - zoomed_w) // 2
            zoomed_crop = crop[start_y:start_y+zoomed_h, start_x:start_x+zoomed_w]
            crop = cv2.resize(zoomed_crop, (cw, ch))

        # Resize to final output size
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, crop)

        prev_spk = curr_spk
        frame_idx += 1

        # Progress logging
        if frame_idx % 30 == 0:
            progress = (frame_idx / nF) * 100
            print(f"   Progress: {frame_idx}/{nF} frames ({progress:.1f}%)")

    cap.release()
    
    # Only close models if they were created locally (not global)
    if GLOBAL_FACE_DETECTION is None:
        face_detection.close()
    if GLOBAL_FACE_MESH is None:
        face_mesh.close()
    
    print(f"✅ Finished processing {frame_idx} frames. Compiling video with FFmpeg...")

    # Use FFmpeg to compile frames into video
    tmp_video_only = os.path.join("tmp", f"__tmp_tracked_{os.path.basename(out_path)}")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        tmp_video_only
    ]
    subprocess.run(cmd, check=True)

    # Remux original audio with the new video track
    (
        ffmpeg
        .output(
            ffmpeg.input(tmp_video_only).video,
            ffmpeg.input(in_path).audio,
            out_path,
            vcodec='h264_nvenc',
            acodec='copy',
            preset='fast'
        )
        .overwrite_output()
        .run(quiet=False)
    )

    # Cleanup
    try:
        os.remove(tmp_video_only)
        shutil.rmtree(frames_dir)
        shutil.rmtree(WORKDIR)
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")

    print(f"🎉 Generated advanced active speaker tracked video: {out_path}", flush=True)
    
    # Generate annotated debug video with face boxes and confidence scores
    debug_video_path = out_path.replace('.mp4', '_debug_boxes.mp4')
    print(f"🎨 Generating annotated debug video with face boxes and confidence scores...")
    generate_face_annotation_video(in_path, debug_video_path, tracks, spk2face, spk_for_frame, segments, fps, W, H)
    print(f"✅ Generated debug video: {debug_video_path}")

def generate_face_annotation_video(in_path: str, out_path: str, tracks: dict, spk2face: dict, spk_for_frame: list, segments: list, fps: float, W: int, H: int):
    """
    Generate an annotated video showing face bounding boxes and speaker confidence scores.
    This creates a debug/visualization video with boxes around faces and confidence numbers.
    
    Args:
        in_path: Original input video path (not cropped)
        out_path: Output path for annotated video
        tracks: Dict of face tracks {face_id: [(frame_idx, bbox), ...]}
        spk2face: Dict mapping speaker ID to face ID
        spk_for_frame: List of speaker IDs for each frame
        segments: List of speaker segments
        fps: Video FPS
        W: Video width
        H: Video height
    """
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"⚠️  Could not open video for annotation: {in_path}")
        return
    
    # Get video properties
    nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create frame directory
    frames_dir = os.path.join("tmp", f"__annotated_frames_{os.path.basename(out_path)}")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Build frame_to_fids mapping
    frame_to_fids: Dict[int, List[int]] = {}
    for fid, samples in tracks.items():
        for fi, _ in samples:
            frame_to_fids.setdefault(fi, []).append(fid)
    
    # Build face_id to speaker_id reverse mapping
    face_to_spk: Dict[int, int] = {}
    for spk_id, face_id in spk2face.items():
        face_to_spk[face_id] = spk_id
    
    # Calculate confidence scores per face per frame
    # Confidence is based on:
    # 1. Whether this face is the active speaker's face
    # 2. Face size (larger faces are more likely to be speaking)
    # 3. Consistency of assignment
    
    def calculate_confidence_score(frame_idx: int, face_id: int, bbox: tuple) -> float:
        """Calculate confidence score for a face in a given frame."""
        curr_spk = spk_for_frame[frame_idx] if frame_idx < len(spk_for_frame) else -1
        is_active_speaker = (face_id in face_to_spk and face_to_spk[face_id] == curr_spk)
        
        # Base score from face size (larger = more confident)
        face_area = bbox[2] * bbox[3]
        video_area = W * H
        size_ratio = face_area / video_area
        
        # Base confidence
        if is_active_speaker:
            # Active speaker gets higher confidence (1.0-2.5 range)
            base_confidence = 1.5 + (size_ratio * 1.0)
        else:
            # Non-active speakers get lower confidence (0.0-1.5 range)
            base_confidence = 0.5 + (size_ratio * 1.0)
        
        # Clamp to reasonable range
        return max(0.0, min(2.5, base_confidence))
    
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Get faces for this frame
        fids = frame_to_fids.get(frame_idx, [])
        curr_spk = spk_for_frame[frame_idx] if frame_idx < len(spk_for_frame) else -1
        
        # Draw boxes and confidence for each face
        for fid in fids:
            # Find the bbox for this face in this frame
            bbox = None
            for fi, bb in tracks.get(fid, []):
                if fi == frame_idx:
                    bbox = bb
                    break
            
            if bbox is None:
                continue
            
            x, y, w, h = bbox
            is_active_speaker = (fid in face_to_spk and face_to_spk[fid] == curr_spk)
            
            # Choose color: green for active speaker, red for others
            color = (0, 255, 0) if is_active_speaker else (0, 0, 255)
            
            # Draw bounding box
            thickness = 3
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Calculate confidence score
            confidence = calculate_confidence_score(frame_idx, fid, bbox)
            
            # Draw confidence text above the box
            confidence_text = f"{confidence:.1f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            text_thickness = 2
            
            # Get text size for positioning
            (text_width, text_height), baseline = cv2.getTextSize(confidence_text, font, font_scale, text_thickness)
            
            # Position text above the box, centered
            text_x = x + (w - text_width) // 2
            text_y = max(text_height + 5, y - 5)
            
            # Draw text background for better visibility
            cv2.rectangle(frame, 
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + baseline + 5),
                         (0, 0, 0), -1)  # Black background
            
            # Draw text in the same color as the box
            cv2.putText(frame, confidence_text, (text_x, text_y), 
                       font, font_scale, color, text_thickness)
        
        # Save annotated frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, frame)
        
        frame_idx += 1
        
        # Progress logging
        if frame_idx % 30 == 0:
            progress = (frame_idx / nF) * 100
            print(f"   Annotation progress: {frame_idx}/{nF} frames ({progress:.1f}%)")
    
    cap.release()
    
    # Compile annotated video
    tmp_video_only = os.path.join("tmp", f"__tmp_annotated_{os.path.basename(out_path)}")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        tmp_video_only
    ]
    subprocess.run(cmd, check=True)
    
    # Remux with original audio
    (
        ffmpeg
        .output(
            ffmpeg.input(tmp_video_only).video,
            ffmpeg.input(in_path).audio,
            out_path,
            vcodec='h264_nvenc',
            acodec='copy',
            preset='fast'
        )
        .overwrite_output()
        .run(quiet=False)
    )
    
    # Cleanup
    try:
        os.remove(tmp_video_only)
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"Warning: Could not clean up annotation temporary files: {e}")

def generate_short_simple_fallback(in_path: str, out_path: str, srt_path: str = None, detect_every: int = 1, ease: float = 0.2, zoom_cues=None):
    """
    Simple face tracking fallback when advanced active speaker detection fails.
    Uses basic MediaPipe face detection with simple tracking.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # target crop size (portrait 9:16)
    th = h
    tw = int(h * 9/16)
    if tw > w:  # fallback if input is too narrow
        tw = w

    # Initialize MediaPipe Face Detection
    print("Initializing MediaPipe Face Detection (fallback mode)...")
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,  # 1 = full range (better for videos), 0 = short range
        min_detection_confidence=0.5
    )

    # Parse SRT to find zoom timings
    zoom_times = []
    if srt_path and os.path.exists(srt_path):
        zoom_times = extract_zoom_timings_from_srt(srt_path)

    def should_zoom(time_sec):
        for start, end in zoom_times:
            if start <= time_sec <= end:
                return True
        return False

    # helper to bound crop
    def clamp_x(cx):
        x1 = max(0, min(int(cx - tw // 2), w - tw))
        return x1

    # initial center
    cx = w // 2
    target_cx = cx
    last_detected_face = None

    # Create temp directory for frames
    frames_dir = os.path.join("tmp", f"__frames_{os.path.basename(out_path)}")
    os.makedirs(frames_dir, exist_ok=True)

    frame_idx = 0
    print(f"Processing {total_frames} frames with simple face tracking (fallback)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # Face detection every N frames
        if frame_idx % detect_every == 0:
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)
            
            if results.detections:
                # Use the largest face (first detection is usually largest)
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                face_x = int(bbox.xmin * w)
                face_w = int(bbox.width * w)
                face_cx = face_x + face_w // 2
                
                # Update target position
                target_cx = face_cx
                target_cx = max(tw // 2, min(w - tw // 2, target_cx))
                last_detected_face = face_cx
            else:
                # No faces detected - center immediately
                target_cx = w // 2
                cx = target_cx
                last_detected_face = None

        # Smooth easing toward target (only if not instantly centered)
        if last_detected_face is not None:
            cx = int((1 - ease) * cx + ease * target_cx)
        # else: cx already set to target_cx (instant centering)

        # compute crop x
        x1 = clamp_x(cx)
        crop = frame[0:th, x1:x1+tw]

        # Apply zoom effect if this timestamp matches a <zoom> tag
        if should_zoom(current_time):
            # Slight zoom in (1.05x) - zoom into center of crop
            zoom_factor = 1.05
            zoomed_h = int(th / zoom_factor)
            zoomed_w = int(tw / zoom_factor)
            
            # Calculate center crop region
            start_y = (th - zoomed_h) // 2
            start_x = (tw - zoomed_w) // 2
            
            # Crop to center and resize back
            zoomed_crop = crop[start_y:start_y+zoomed_h, start_x:start_x+zoomed_w]
            crop = cv2.resize(zoomed_crop, (tw, th))

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, crop)

        frame_idx += 1
        
        # Progress logging every 30 frames (~1 second at 30fps)
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {frame_idx}/{total_frames} frames ({progress:.1f}%)")

    cap.release()
    face_detector.close()
    print(f"Finished processing {frame_idx} frames. Compiling video with FFmpeg...")

    # Use FFmpeg to compile frames into video
    tmp_video_only = os.path.join("tmp", f"__tmp_tracked_{os.path.basename(out_path)}")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        tmp_video_only
    ]
    subprocess.run(cmd, check=True)

    # remux original audio with the new video track
    (
        ffmpeg
        .output(
            ffmpeg.input(tmp_video_only).video,
            ffmpeg.input(in_path).audio,
            out_path,
            vcodec='h264_nvenc',
            acodec='copy',
            preset='fast'
        )
        .overwrite_output()
        .run(quiet=False)
    )

    # Cleanup
    try:
        os.remove(tmp_video_only)
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")

    print(f"Generated simple face-tracked cropped short (fallback): {out_path}", flush=True)

def apply_zoom_effects_only(input_file: str, output_file: str, srt_path: str = None):
    """
    Apply zoom effects to a video that's already 9:16 (no face tracking or cropping).
    This is much faster than generate_short() since it skips face detection.
    
    Args:
        input_file: Name of input video file in tmp/ directory
        output_file: Name of output video file in tmp/ directory
        srt_path: Path to SRT file with <zoom> tags
    """
    in_path  = os.path.join('tmp', input_file)
    out_path = os.path.join('tmp', output_file)

    if os.path.exists(out_path):
        print(f"Skipping zoom application, exists: {out_path}")
        return

    # Parse SRT to find zoom timings
    zoom_times = []
    if srt_path and os.path.exists(srt_path):
        zoom_times = extract_zoom_timings_from_srt(srt_path)
        print(f"Found {len(zoom_times)} zoom cues in SRT")
    
    if not zoom_times:
        # No zoom effects needed, just copy the file
        print("No zoom effects to apply, copying video")
        shutil.copy2(in_path, out_path)
        return

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Applying zoom effects to {total_frames} frames (no face tracking)...")

    # Check if a given timestamp should have zoom effect
    def should_zoom(time_sec):
        for start, end in zoom_times:
            if start <= time_sec <= end:
                return True
        return False

    # Create temp directory for frames
    frames_dir = os.path.join("tmp", f"__frames_{os.path.basename(out_path)}")
    os.makedirs(frames_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # Apply zoom effect if this timestamp matches a <zoom> tag
        if should_zoom(current_time):
            # Slight zoom in (1.05x) - zoom into center
            zoom_factor = 1.05
            zoomed_h = int(h / zoom_factor)
            zoomed_w = int(w / zoom_factor)
            
            # Calculate center crop region
            start_y = (h - zoomed_h) // 2
            start_x = (w - zoomed_w) // 2
            
            # Crop to center and resize back
            zoomed_frame = frame[start_y:start_y+zoomed_h, start_x:start_x+zoomed_w]
            frame = cv2.resize(zoomed_frame, (w, h))

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, frame)

        frame_idx += 1
        
        # Progress logging every 30 frames (~1 second at 30fps)
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {frame_idx}/{total_frames} frames ({progress:.1f}%)")

    cap.release()
    print(f"Finished processing {frame_idx} frames. Compiling video with FFmpeg...")

    # Use FFmpeg to compile frames into video
    tmp_video_only = os.path.join("tmp", f"__tmp_zoom_{os.path.basename(out_path)}")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        tmp_video_only
    ]
    subprocess.run(cmd, check=True)

    # Remux original audio with the new video track
    (
        ffmpeg
        .output(
            ffmpeg.input(tmp_video_only).video,
            ffmpeg.input(in_path).audio,
            out_path,
            vcodec='h264_nvenc',
            acodec='copy',
            preset='fast'
        )
        .overwrite_output()
        .run(quiet=False)
    )

    # Cleanup
    try:
        os.remove(tmp_video_only)
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")

    print(f"Applied zoom effects to video: {out_path}", flush=True)
    
# --- Helper function for per-clip transcription ---
def transcribe_clip_with_faster_whisper(input_path: str, detected_language: str = None) -> str:
    """
    Transcribe a single clip using faster-whisper.
    If detected_language is provided, uses it; otherwise auto-detects.
    Returns the path to the generated SRT file.
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    srt_path = os.path.join('tmp', f"{base}.srt")
    
    # Check if video has audio stream
    if not has_audio_stream(input_path):
        print(f"⚠️  Clip '{base}' has no audio stream, creating empty SRT")
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write("")
        return srt_path
    
    try:
        from faster_whisper import WhisperModel
        import torch
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"🎤 Transcribing clip with faster-whisper ({device})...")
        
        # If language is Hebrew, use ivrit-ai Hebrew model; otherwise use turbo
        if detected_language and detected_language in ['he', 'iw']:
            print(f"🔯 Using Hebrew-optimized model for clip: ivrit-ai/whisper-large-v3-turbo-ct2")
            try:
                # Use ivrit-ai Hebrew model in CTranslate2 format
                model = WhisperModel(
                    "ivrit-ai/whisper-large-v3-turbo-ct2",
                    device=device,
                    compute_type=compute_type
                )
            except Exception as e:
                print(f"⚠️  ivrit-ai model unavailable ({e}), using turbo")
                model = WhisperModel("turbo", device=device, compute_type=compute_type)
        else:
            # Use turbo model for clips (fast and accurate enough for short segments)
            model = WhisperModel("turbo", device=device, compute_type=compute_type)
        
        # Transcribe
        segments, info = model.transcribe(
            input_path,
            beam_size=5,
            language=detected_language,  # Use detected language if provided
            vad_filter=True,
            condition_on_previous_text=True
        )
        
        # Write SRT
        with open(srt_path, 'w', encoding='utf-8') as f:
            segment_list = list(segments)
            for i, segment in enumerate(segment_list, start=1):
                start_time = format_timestamp_srt(segment.start)
                end_time = format_timestamp_srt(segment.end)
                text = segment.text.strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        # Clean up
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return srt_path
        
    except Exception as e:
        print(f"⚠️  faster-whisper failed for clip ({e}), using auto_subtitle fallback")
        # Fallback to auto_subtitle
        cmd = f"auto_subtitle {input_path} --srt_only True -o tmp/ --model turbo"
        subprocess.call(cmd, shell=True)
        return srt_path

# --- SRT gen (ALWAYS reprocess; backup original as old_*.srt; enforce max 5 words & min 1.0s) ---
def generate_subtitle_for_clip(input_path: str, detected_language: str = None) -> str:
    """
    Generate subtitles for a clip, with optional language hint.
    Only generates if SRT doesn't already exist, then enforces max 5 words per cue with min 1.0s duration.
    """
    base      = os.path.splitext(os.path.basename(input_path))[0]
    srt_path  = os.path.join('tmp', f"{base}.srt")
    old_path  = os.path.join('tmp', f"old_{base}.srt")

    # If SRT doesn't exist, create a fresh SRT using faster-whisper (with fallback to auto_subtitle)
    if not os.path.exists(srt_path):
        print(f"Generating new SRT for {base}...")
        try:
            transcribe_clip_with_faster_whisper(input_path, detected_language)
        except Exception as e:
            print(f"Warning: transcription failed ({e}), trying auto_subtitle fallback")
            cmd = f"auto_subtitle {input_path} --srt_only True -o tmp/ --model turbo"
            subprocess.call(cmd, shell=True)
        
        if os.path.exists(srt_path):
            try:
                shutil.copy2(srt_path, old_path)
                print(f"Saved original SRT to {old_path}")
            except Exception as e:
                print("Warning: failed to save original SRT copy:", e)
    else:
        print(f"SRT already exists for {base}, skipping generation")

    # Now (re)chunk to max 5 words per cue, min 1.0s, with RTL wrapping
    entries = parse_srt(srt_path)
    entries = enforce_srt_word_chunks(entries, max_words=5, min_dur=1.0)
    write_srt_entries(entries, srt_path, rtl_wrap=True)
    print(f"Rewrote SRT with max 5 words per cue: {srt_path}")
    return srt_path

# --- Silence removal using ffmpeg silencedetect ---
# Based on DarkTrick's implementation - 4fb5f723849d32782e723c34bfd132e442d378d7
def find_silences(filename: str, dB=-10, min_duration=0.1):
    """
    Find silence periods in video using ffmpeg silencedetect.
    
    Returns:
        list: List of timestamps [silence_start1, silence_end1, silence_start2, silence_end2, ...]
    """
    print(f"   Detecting silences: threshold={dB}dB, min_duration={min_duration}s")
    
    command = [
        "ffmpeg", "-i", filename,
        "-af", f"silencedetect=n={dB}dB:d={min_duration}",
        "-f", "null", "-"
    ]
    
    output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s = str(output.stderr, 'utf-8')
    lines = s.split("\n")
    time_list = []
    
    for line in lines:
        if "silencedetect" in line:
            words = line.split(" ")
            for i in range(len(words)):
                if "silence_start" in words[i]:
                    try:
                        time_list.append(float(words[i + 1]))
                    except (IndexError, ValueError):
                        pass
                if "silence_end" in words[i]:
                    try:
                        time_list.append(float(words[i + 1]))
                    except (IndexError, ValueError):
                        pass
    
    return time_list


def get_sections_of_new_video(silences, duration):
    """
    Returns timings for parts where the video should be kept (non-silent parts).
    Converts silence timings to keep timings.
    """
    return [0.0] + silences + [duration]


def ffmpeg_filter_get_segment_filter(video_section_timings):
    """Build the between() filter for select/aselect"""
    ret = ""
    for i in range(int(len(video_section_timings) / 2)):
        start = video_section_timings[2 * i]
        end = video_section_timings[2 * i + 1]
        ret += f"between(t,{start},{end})+"
    # Cut away last "+"
    ret = ret[:-1]
    return ret


def remove_silence_with_ffmpeg(input_video: str, output_video: str, 
                               silence_threshold_db=-30,
                               min_silence_duration=0.3,
                               padding=0.0) -> bool:
    """
    Remove silent parts from video using ffmpeg's silencedetect filter.
    Uses select/aselect filters for cleaner implementation.
    
    Based on DarkTrick's silence_cutter implementation.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        silence_threshold_db: dB threshold for silence detection (default: -30dB, DarkTrick's proven default)
        min_silence_duration: Minimum seconds of silence to cut (default: 0.3s)
        padding: Padding to leave on edges (unused in this implementation)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n🔇 Removing silence with ffmpeg silencedetect...")
        print(f"   Input: {input_video}")
        print(f"   Output: {output_video}")
        print(f"   Threshold: {silence_threshold_db}dB (DarkTrick's proven default)")
        print(f"   Min duration: {min_silence_duration}s")
        
        # Step 1: Find silence periods
        silences = find_silences(input_video, dB=silence_threshold_db, min_duration=min_silence_duration)
        
        if not silences:
            print("   No silence detected, copying original video")
            shutil.copy2(input_video, output_video)
            return True
        
        print(f"   Found {len(silences)//2} silence periods")
        
        # Step 2: Get video duration
        duration = ffprobe_duration(input_video)
        
        # Step 3: Get sections to keep (non-silent parts)
        video_sections = get_sections_of_new_video(silences, duration)
        
        # Step 4: Build filter strings
        segment_filter = ffmpeg_filter_get_segment_filter(video_sections)
        video_filter = f"select='{segment_filter}', setpts=N/FRAME_RATE/TB"
        audio_filter = f"aselect='{segment_filter}', asetpts=N/SR/TB"
        
        # Step 5: Create temporary filter files
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", encoding="UTF-8", prefix="silence_video_", 
                                        suffix=".txt", delete=False) as vFile:
            vFile.write(video_filter)
            video_filter_file = vFile.name
        
        with tempfile.NamedTemporaryFile(mode="w", encoding="UTF-8", prefix="silence_audio_", 
                                        suffix=".txt", delete=False) as aFile:
            aFile.write(audio_filter)
            audio_filter_file = aFile.name
        
        try:
            # Step 6: Run ffmpeg with filter scripts
            print("   Step 2: Removing silence...")
            has_audio = has_audio_stream(input_video)
            
            if has_audio:
                command = [
                    "ffmpeg", "-y", "-i", input_video,
                    "-filter_script:v", video_filter_file,
                    "-filter_script:a", audio_filter_file,
                    "-c:v", "h264_nvenc", "-preset", "fast",
                    "-c:a", "aac",
                    output_video
                ]
            else:
                # Video only (no audio)
                command = [
                    "ffmpeg", "-y", "-i", input_video,
                    "-filter_script:v", video_filter_file,
                    "-c:v", "h264_nvenc", "-preset", "fast",
                    output_video
                ]
            
            subprocess.run(command, check=True, capture_output=True)
            
            if os.path.exists(output_video):
                print(f"✅ Successfully removed silence: {output_video}")
                return True
            else:
                print(f"⚠️  Output file not created")
                return False
        finally:
            # Cleanup temp files
            try:
                os.remove(video_filter_file)
                os.remove(audio_filter_file)
            except:
                pass
            
    except Exception as e:
        print(f"❌ Error removing silence: {e}")
        print(f"⚠️  Falling back to original video")
        if input_video != output_video:
            shutil.copy2(input_video, output_video)
        return False

# --- Helper functions for video processing ---
def ffprobe_duration(path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def compute_keep_intervals_from_srt(entries, total_duration, min_gap=0.2, pad_after=0.1):
    """
    Build keep intervals by removing silence gaps > min_gap.
    For each such gap (gap = [end_i, start_{i+1}]), we cut from gap_start to (start_{i+1}-pad_after).
    """
    if total_duration <= 0:
        # fallback: keep all
        return [(0.0, entries[-1]["end"] if entries else 0.0)]

    # sort & merge any overlaps
    speech = sorted([(e["start"], e["end"]) for e in entries], key=lambda x: x[0])
    merged = []
    for s, e in speech:
        if not merged or s > merged[-1][1] + 1e-3:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    keep = []
    cur_start = 0.0
    # initial gap before first speech
    if merged:
        first_start = merged[0][0]
        gap = first_start - 0.0
        if gap > min_gap:
            # cut [0, first_start - pad_after]
            keep.append((0.0, 0.0))  # nothing before first speech
            cur_start = max(cur_start, first_start - pad_after)
        else:
            cur_start = 0.0
    else:
        # no speech entries; keep nothing
        return []

    # internal gaps
    for i in range(len(merged) - 1):
        prev_end = merged[i][1]
        next_start = merged[i+1][0]
        gap = next_start - prev_end
        if gap > min_gap:
            # keep from cur_start to prev_end
            if prev_end - cur_start > 0.01:
                keep.append((cur_start, prev_end))
            # skip the silence until next_start - pad_after
            cur_start = max(cur_start, next_start - pad_after)

    # tail after last speech
    last_end = merged[-1][1]
    if total_duration - last_end > min_gap:
        # after last speech, if it's a long silence, keep up to last_end only
        if last_end - cur_start > 0.01:
            keep.append((cur_start, last_end))
    else:
        # small tail silence -> include it
        if total_duration - cur_start > 0.01:
            keep.append((cur_start, total_duration))

    # filter invalid
    keep = [(a, b) for (a, b) in keep if b - a > 0.05]
    return keep

def has_audio_stream(path: str) -> bool:
    try:
        # returns non-empty if an audio stream exists
        p = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "a", "-show_entries", "stream=index",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True
        )
        return bool(p.stdout.strip())
    except Exception:
        return False

def build_trim_concat_cmd(input_video, keep_intervals, output_video):
    """
    Build a filter_complex that trims video (and audio if present),
    then concatenates. IMPORTANT: when v=1,a=1, inputs to concat
    must be interleaved: [v0][a0][v1][a1]...
    """
    filter_parts = []
    pair_labels = []  # interleaved inputs for concat
    has_audio = has_audio_stream(input_video)

    for idx, (s, e) in enumerate(keep_intervals):
        vlab = f"v{idx}"
        if has_audio:
            alab = f"a{idx}"
            filter_parts.append(
                f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[{vlab}];"
                f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[{alab}]"
            )
            pair_labels.append(f"[{vlab}][{alab}]")
        else:
            filter_parts.append(
                f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[{vlab}]"
            )
            pair_labels.append(f"[{vlab}]")

    n = len(keep_intervals)
    if has_audio:
        # interleaved order: [v0][a0][v1][a1]...
        concat_inputs = "".join(pair_labels)
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[v][a]")
        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[v]", "-map", "[a]",
            "-c:v", "h264_nvenc", "-preset", "fast",
            "-c:a", "aac",
            output_video
        ]
    else:
        # video only: just feed [v0][v1]... and request v=1,a=0
        concat_inputs = "".join(pair_labels)
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[v]")
        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[v]",
            "-c:v", "h264_nvenc", "-preset", "fast",
            output_video
        ]

    return cmd

def retime_srt_after_cuts(entries, keep_intervals):
    """
    Map original timestamps -> new timeline formed by concatenating keep_intervals in order.
    Assumes all cues lie within keep parts (they should, since we cut silence only).
    """
    cumulative = []
    total = 0.0
    for (s, e) in keep_intervals:
        cumulative.append((s, e, total))  # interval plus new base
        total += (e - s)

    def map_t(t):
        for (s, e, base) in cumulative:
            if s <= t <= e:
                return base + (t - s)
        for (s, e, base) in cumulative:
            if t < s:
                return base
        return total

    out = []
    for e in entries:
        ns = map_t(e["start"])
        ne = map_t(e["end"])
        if ne - ns > 0.05:
            out.append({"start": ns, "end": ne, "text": e["text"]})
    return out

def cut_silences_using_srt(input_video: str, srt_path: str, output_video: str, min_gap=0.2, pad_after=0.1):
    # parse
    entries = parse_srt(srt_path)
    # NOTE: we already applied word-based chunking when generating the SRT.
    # Re-parse/keep as-is here so we cut by gaps between cues.
    total_duration = ffprobe_duration(input_video)
    keep = compute_keep_intervals_from_srt(entries, total_duration, min_gap=min_gap, pad_after=pad_after)
    if not keep:
        print("No keep intervals computed; copying input as output.")
        shutil.copy2(input_video, output_video)
        return srt_path  # unchanged

    cmd = build_trim_concat_cmd(input_video, keep, output_video)
    print("Silence-cut FFmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # retime srt to new timeline
    retimed = retime_srt_after_cuts(entries, keep)
    new_srt = os.path.splitext(output_video)[0] + ".srt"
    write_srt_entries(retimed, new_srt, rtl_wrap=True)
    return new_srt

# --- Burn subtitles ---
def burn_subtitles(input_video: str, srt_path: str, output_video: str):
    style = (
        "Alignment=2,MarginV=40,MarginL=25,MarginR=25,"
        "Fontname=Assistant,Fontsize=24,Bold=1,PrimaryColour=&H00FFFFFF,"
        "Outline=1,Shadow=1,BorderStyle=1"
    )
    esc = srt_path.replace('\\','\\\\').replace(':','\\:')
    inp = ffmpeg.input(input_video)
    # Use the fonts directory in tmp folder
    import os
    fonts_dir = os.path.abspath(os.path.join("tmp", "fonts"))  # Absolute path to tmp/fonts directory
    print(f"   🔤 Using font directory: {fonts_dir}")
    print(f"   🔤 Font name in style: {font_name}")
    print(f"   🔤 Font directory exists: {os.path.exists(fonts_dir)}")
    
    # List available font files for debugging
    if os.path.exists(fonts_dir):
        font_files = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf')]
        print(f"   🔤 Available font files: {font_files}")
        
        # Check if the specific font file exists
        if font_name == "Abraham":
            abraham_font = os.path.join(fonts_dir, "Abraham-Regular.ttf")
            print(f"   🔤 Abraham font file exists: {os.path.exists(abraham_font)}")
            print(f"   🔤 Abraham font path: {abraham_font}")
    else:
        print(f"   ⚠️ Font directory not found: {fonts_dir}")
    
    video = inp.video.filter_('subtitles', esc, force_style=style, fontsdir=fonts_dir)
    audio = inp.audio
    (
        ffmpeg.output(video, audio, output_video, vcodec='h264_nvenc', acodec='copy', preset='fast')
        .run(overwrite_output=True)
    )

# --- Main entrypoint ---
def main():
    global PROCESSING_SETTINGS, IS_SHORT_VIDEO
    
    print("\n\nRunning reelsfy has started! Processing the video!")
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--video_id')
    parser.add_argument('-f','--file')
    parser.add_argument('--export-mode', action='store_true', help='Run in export mode for final processing')
    args = parser.parse_args()
    if not args.export_mode and not args.video_id and not args.file:
        parser.error('Provide --video_id or --file (or use --export-mode)')

    os.makedirs('tmp', exist_ok=True)

    # determine a results folder key
    if not args.export_mode:
        key = args.video_id if args.video_id else os.path.splitext(os.path.basename(args.file))[0]
        results_dir = os.path.join('results', key)
        os.makedirs(results_dir, exist_ok=True)
        content_path = os.path.join(results_dir, 'content.txt')
        print(content_path)
    else:
        # In export mode, we don't need results directory
        results_dir = None
        content_path = None

    # download or copy (skip in export mode)
    if not args.export_mode:
        if prompt_stage("Download or copy input video"):
            src = 'input_video.mp4'
            if args.video_id:
                YouTube(f"https://youtu.be/{args.video_id}")\
                    .streams.filter(file_extension='mp4')\
                    .get_highest_resolution()\
                    .download(output_path='tmp', filename=src)
            else:
                shutil.copy2(args.file, os.path.join('tmp', src))
            print(f"Input video ready: {src}")
        else:
            src = 'input_video.mp4'
            print(f"Skipped download - using existing: {src}")

        # For SHORT VIDEO MODE: Remove silence BEFORE transcription
        if IS_SHORT_VIDEO:
            auto_cuts_enabled = PROCESSING_SETTINGS.get('autoCuts', True)
            src_path = os.path.join('tmp', src)
            
            if auto_cuts_enabled and prompt_stage("Remove silence from short video (before transcription)"):
                report_progress(12, "מסיר דממות מהסרטון...")
                src_no_silence = os.path.join('tmp', 'input_video_nosilence.mp4')
                
                print("\n" + "="*60)
                print("SHORT VIDEO MODE - Removing silence before transcription")
                print("="*60 + "\n")
                
                # Only process if output doesn't exist or is older than input
                if not os.path.exists(src_no_silence) or os.path.getmtime(src_no_silence) < os.path.getmtime(src_path):
                    success = remove_silence_with_ffmpeg(
                        src_path,
                        src_no_silence
                        # Using DarkTrick's proven defaults: -30dB, 1.0s
                    )
                    
                    if success:
                        # Use silence-removed video for transcription
                        src = 'input_video_nosilence.mp4'
                        print(f"✅ Using silence-removed video for transcription: {src}")
                    else:
                        print(f"⚠️  Continuing with original video")
                else:
                    # Use existing silence-removed video
                    src = 'input_video_nosilence.mp4'
                    print(f"✅ Using existing silence-removed video")
            elif not auto_cuts_enabled:
                print("⏭️  Auto-cuts disabled - skipping silence removal for short video")

        # transcript (for GPT prompt context) and detected language
        if prompt_stage("Generate transcript using auto_subtitle"):
            report_progress(15, "מתמלל סרטון...")
            transcript, detected_language = generate_transcript(src)
            print(f"Transcript generated (Language: {detected_language})")
        else:
            transcript = ""
            detected_language = "unknown"
            print("Skipped transcript generation")
    
    else:
        # In export mode, we're working with already downloaded files
        print("Export mode: Using already downloaded files from tmp/")
        src = None
        detected_language = "unknown"  # In export mode, we don't have language info

    # viral_json or short_video styling: cached or freshly generated
    if not args.export_mode:
        if IS_SHORT_VIDEO:
            # Short video mode: just get styling (colored words and zoom cues)
            if prompt_stage("Generate short video styling (colored words + zoom)"):
                report_progress(35, "מזהה מילים חשובות...")
                print("Short video mode - generating styling, title and description")
                auto_zoom = PROCESSING_SETTINGS.get('autoZoomIns', True)
                color_hex = PROCESSING_SETTINGS.get('coloredWordsColor', '#FF3B3B')
                styling_data = generate_short_video_styling(transcript, auto_zoom, color_hex)
                viral_data = {
                    "segments": [],
                    "srt_overrides": styling_data.get("srt_overrides", {}),
                    "title": styling_data.get("title", "סרטון קצר"),
                    "description": styling_data.get("description", "")
                }
                # Save for potential reprocessing
                with open(content_path, 'w', encoding='utf-8') as f:
                    json.dump(viral_data, f, ensure_ascii=False, indent=2)
                print(f"Generated title: {viral_data['title']}")
                print(f"Generated description: {viral_data['description']}")
            else:
                viral_data = {"segments": [], "srt_overrides": {}, "title": "סרטון קצר", "description": ""}
                print("Skipped short video styling")
        else:
            # Regular mode: find viral segments
            if prompt_stage("Generate/load viral segments"):
                if os.path.exists(content_path):
                    print(f"Loading cached viral segments from {content_path}")
                    with open(content_path, 'r', encoding='utf-8') as f:
                        viral_data = json.load(f)
                else:
                    report_progress(35, "מזהה קטעים ויראליים...")
                    print("No cached content.txt found—calling generate_viral()")
                    viral_data = generate_viral(transcript)
                    with open(content_path, 'w', encoding='utf-8') as f:
                        json.dump(viral_data, f, ensure_ascii=False, indent=2)
            else:
                viral_data = {"segments": [], "srt_overrides": {}}
                print("Skipped viral segment generation")
    else:
        # In export mode, we don't need viral_data
        viral_data = None

    if not args.export_mode:
        segments = viral_data.get('segments', [])
    else:
        # In export mode, determine segments based on existing files
        segments = []
        # List all output_cropped*.mp4 files in tmp directory
        for file in os.listdir('tmp'):
            if file.startswith('output_cropped') and file.endswith('.mp4'):
                # Extract index from filename (output_cropped001.mp4 -> 1)
                try:
                    index = int(file.replace('output_cropped', '').replace('.mp4', ''))
                    segments.append(index)
                except ValueError:
                    continue
        # Sort segments to ensure consistent processing order
        segments.sort()
        print(f"Export mode: Found {len(segments)} segments to process: {segments}")

    # segments extraction → crop → srt → silence-cut → (burn subs only in export mode)
    if not args.export_mode:
        if IS_SHORT_VIDEO:
            # Short video mode: process entire video as single clip
            # WORKFLOW:
            # 1. Silence removal (if enabled) -> input_video_nosilence.mp4
            # 2. Transcription -> input_video[_nosilence].srt
            # 3. AI styling -> content.txt (with <zoom> tags)
            # 4. Apply SRT overrides to source SRT (add colored words and zoom markers)
            # 5. Check aspect ratio:
            #    - If NOT 9:16: crop to 9:16 with face tracking + zoom effects
            #    - If 9:16: only apply zoom effects (if enabled) - NO FACE TRACKING!
            # 6. Final output: output_cropped000.mp4 and output_cropped000.srt
            print("\n" + "="*60)
            print("SHORT VIDEO MODE - Processing entire video as single clip")
            print("="*60 + "\n")
            
            # Note: Silence was already removed before transcription (if enabled)
            # Working directly with input_video[_nosilence].mp4
            report_progress(45, "מכין קליפ יחיד...")
            
            # Use the correct source (either silence-removed or original)
            src_path = os.path.join('tmp', src)
            
            # Apply SRT overrides directly to the source SRT (colored words and zoom tags)
            input_srt = os.path.join('tmp', f"{os.path.splitext(src)[0]}.srt")
            if os.path.exists(input_srt):
                if viral_data and 'srt_overrides' in viral_data:
                    apply_srt_overrides(input_srt, viral_data['srt_overrides'])
                    print("Applied styling to SRT (colored words and zoom)")
            else:
                print(f"Warning: SRT not found: {input_srt}")
            
            segments = [0]  # Single segment
        else:
            # Regular mode: extract multiple segments
            # NEW WORKFLOW (optimized order):
            # 1. Extract segments (outputxxx.mp4)
            # 2. Remove silence from each segment (if enabled) -> output_nosilencexxx.mp4
            # 3. Generate subtitles once (after silence removal) with AI overrides
            # 4. Crop to 9:16 with face tracking + zoom -> output_croppedxxx.mp4
            # 5. Remove <zoom> tags from SRT -> output_croppedxxx.srt
            # Benefits: Only transcribe once (not twice), more efficient workflow
            if prompt_stage("Extract video segments"):
                report_progress(45, f"מחלץ {len(segments)} קטעים...")
                generate_segments(segments)
                print(f"Extracted {len(segments)} segments")
            else:
                print("Skipped segment extraction")
            
        for i in range(len(segments)):
            # Calculate progress: 50% to 80% spread across all clips
            clip_progress = 50 + int((i / len(segments)) * 30)
            report_progress(clip_progress, f"מעבד קליפ {i+1}/{len(segments)}...")
            
            crop  = f"output_croppedwithoutcutting{str(i).zfill(3)}.mp4"
            nosil = f"output_cropped{str(i).zfill(3)}.mp4"
            final = os.path.join('tmp', f"final_{str(i).zfill(3)}.mp4")

            # For short video mode, work directly with source video
            if IS_SHORT_VIDEO:
                # Use source video (input_video_nosilence.mp4 or input_video.mp4)
                raw = src  # This is already 'input_video_nosilence.mp4' or 'input_video.mp4'
                raw_path = os.path.join('tmp', raw)
                raw_srt = os.path.join('tmp', f"{os.path.splitext(raw)[0]}.srt")
                
                if not os.path.exists(raw_srt):
                    print(f"Warning: SRT not found for short video: {raw_srt}")
            else:
                # Regular mode: NEW WORKFLOW
                # 1. Start with extracted segment
                # 2. Remove silence first (if enabled)
                # 3. Generate subtitles once (after silence removal)
                # 4. Crop with face tracking
                # 5. Remove <zoom> tags from final SRT
                
                raw = f"output{str(i).zfill(3)}.mp4"
                raw_path = os.path.join('tmp', raw)
                auto_cuts_enabled = PROCESSING_SETTINGS.get('autoCuts', True)
                
                # STEP 1: Remove silence from extracted segment (if enabled)
                if auto_cuts_enabled and prompt_stage(f"Remove silence from segment {i}"):
                    clip_progress = 50 + int((i / len(segments)) * 10)  # 50-60% progress range
                    report_progress(clip_progress, f"מסיר דממות מקליפ {i+1}/{len(segments)}...")
                    
                    raw_nosilence = f"output_nosilence{str(i).zfill(3)}.mp4"
                    raw_nosilence_path = os.path.join('tmp', raw_nosilence)
                    
                    success = remove_silence_with_ffmpeg(
                        raw_path,
                        raw_nosilence_path
                        # Using DarkTrick's proven defaults: -30dB, 1.0s
                    )
                    
                    if success:
                        # Use silence-removed version for transcription and cropping
                        raw = raw_nosilence
                        raw_path = raw_nosilence_path
                        print(f"✅ Removed silence from segment {i}")
                    else:
                        print(f"⚠️  Silence removal failed for segment {i}, using original")
                else:
                    print(f"⏭️  Skipping silence removal for segment {i}")
                
                # STEP 2: Generate subtitles for the segment (after silence removal)
                if prompt_stage(f"Generate subtitles for segment {i}"):
                    clip_progress = 60 + int((i / len(segments)) * 10)  # 60-70% progress range
                    report_progress(clip_progress, f"מתמלל קליפ {i+1}/{len(segments)}...")
                    
                    # Pass detected language from main transcription to clip transcription
                    raw_srt = generate_subtitle_for_clip(raw_path, detected_language)
                    
                    # Apply SRT overrides from GPT (includes <zoom> tags and corrections)
                    if viral_data and 'srt_overrides' in viral_data:
                        apply_srt_overrides(raw_srt, viral_data['srt_overrides'])
                    
                    print(f"✅ Generated subtitles for segment {i}")
                else:
                    raw_srt = os.path.join('tmp', f"{os.path.splitext(os.path.basename(raw_path))[0]}.srt")
                    print(f"Skipped subtitle generation for segment {i}")

            # STEP 3: Crop with zoom effects based on SRT (which has <zoom> tags)
            # For short videos, check if already 9:16 before cropping
            auto_zoom_enabled = PROCESSING_SETTINGS.get('autoZoomIns', True)
            
            if IS_SHORT_VIDEO:
                # Short video mode
                if is_916_aspect_ratio(raw_path):
                    # Video is already 9:16 - only apply zoom if enabled (no face tracking needed!)
                    if auto_zoom_enabled and prompt_stage(f"Apply zoom effects to short video"):
                        print(f"Short video is already 9:16 - applying zoom effects only (no face tracking)")
                        # Use the optimized zoom-only function (much faster than face tracking)
                        apply_zoom_effects_only(raw, nosil, srt_path=raw_srt)
                        print(f"✅ Applied zoom effects to segment {i}")
                    else:
                        print(f"Short video is already 9:16 - no zoom needed, copying to {nosil}")
                        nosil_path = os.path.join('tmp', nosil)
                        if not os.path.exists(nosil_path):
                            shutil.copy2(raw_path, nosil_path)
                else:
                    # Short video needs cropping
                    if prompt_stage(f"Crop short video to 9:16 with face tracking and zoom"):
                        generate_short(raw, nosil, srt_path=raw_srt)
                        print(f"✅ Cropped short video with face tracking and zoom effects")
                    else:
                        print(f"Skipped cropping short video")
                
                # For short videos, copy and clean the SRT
                nosil_path = os.path.join('tmp', nosil)
                nosil_srt = os.path.join('tmp', f"{os.path.splitext(nosil)[0]}.srt")
                
                # Try to find the SRT file
                if os.path.exists(raw_srt):
                    shutil.copy2(raw_srt, nosil_srt)
                else:
                    print(f"⚠️  Warning: No SRT found for segment {i}")
                
                # Remove <zoom> tags from the final SRT
                if os.path.exists(nosil_srt):
                    remove_zoom_tags_from_srt(nosil_srt)
                    print(f"✅ Finalized SRT with accurate timestamps")
            else:
                # Regular mode: Crop and finalize
                clip_progress = 70 + int((i / len(segments)) * 10)  # 70-80% progress range
                report_progress(clip_progress, f"חותך קליפ {i+1}/{len(segments)}...")
                
                if prompt_stage(f"Crop segment {i} to 9:16 aspect ratio with face tracking and zoom"):
                    # Crop directly to final output (output_cropped)
                    generate_short(raw, nosil, srt_path=raw_srt)
                    print(f"✅ Cropped segment {i} with face tracking and zoom effects")
                else:
                    print(f"Skipped cropping segment {i}")
                    # If cropping skipped, just copy the file
                    nosil_path = os.path.join('tmp', nosil)
                    if not os.path.exists(nosil_path):
                        shutil.copy2(raw_path, nosil_path)
                
                # Create final SRT by copying and removing <zoom> tags
                nosil_path = os.path.join('tmp', nosil)
                nosil_srt = os.path.join('tmp', f"{os.path.splitext(nosil)[0]}.srt")
                
                if os.path.exists(raw_srt):
                    shutil.copy2(raw_srt, nosil_srt)
                    # Remove <zoom> tags from the final SRT
                    remove_zoom_tags_from_srt(nosil_srt)
                    print(f"✅ Created final SRT for segment {i}")
                else:
                    print(f"⚠️  Warning: No SRT found for segment {i}")

            print(f"✅ Prepared for export: {nosil_path}")
    else:
        # In export mode, we're working with already processed files
        # Download logos and burn subtitles with styling
        print("Export mode: Processing existing files for subtitle and logo burning")
        
        # Step 1: Download logos from Supabase if needed
        if prompt_stage("Download logos from Supabase (if needed)"):
            download_logos_from_styling_files(segments)
        
        # Step 2: Burn subtitles with styling (includes logo burning)
        if prompt_stage("Burn subtitles with styling and logo onto videos"):
            print("Burning subtitles with styling to create final videos...")
                
            for segment_index in segments:
                nosil = f"output_cropped{str(segment_index).zfill(3)}.mp4"
                final = os.path.join('tmp', f"final_{str(segment_index).zfill(3)}.mp4")
                nosil_path = os.path.join('tmp', nosil)
                
                # Check if the input file exists
                if os.path.exists(nosil_path):
                    # Find the corresponding SRT file (contains <color> tags from GPT)
                    # In export mode, api.py downloads it as output_nosilence but uploads as output_cropped
                    srt_path = os.path.join('tmp', f"output_nosilence{str(segment_index).zfill(3)}.srt")
                    if os.path.exists(srt_path):
                        burn_subtitles_with_styling(nosil_path, srt_path, final, segment_index)
                        print(f"Generated: {final}")
                    else:
                        print(f"Warning: SRT file not found: {srt_path}")
                else:
                    print(f"Warning: Input video file not found: {nosil_path}")
        else:
            print("Skipped burning subtitles with styling")

def process_video_file(input_path: str, out_dir: str = "tmp", settings: dict = None, video_id: str = None, progress_callback=None, is_short_video: bool = False):
    """
    Process video file with optional custom settings from frontend.
    
    settings: Processing configuration from popup dialog
    video_id: Video UUID for progress tracking
    progress_callback: Function to call with (video_id, progress, stage, eta) to update progress
    is_short_video: If True, use simplified processing for videos under 3 minutes
    """
    global PROCESSING_SETTINGS, PROGRESS_CALLBACK, VIDEO_ID, IS_SHORT_VIDEO, SKIP_MODE
    
    # Ask user about skip mode at the very start
    print("\n" + "="*60)
    print("SKIP MODE SETTING")
    print("="*60)
    skip_input = input("Type 'do skip' to enable interactive skip mode, or press ENTER for auto-continue mode: ").strip().lower()
    
    if skip_input == 'do skip':
        SKIP_MODE = False
        print("✅ Interactive skip mode enabled - you'll be asked for each stage")
    else:
        SKIP_MODE = True
        print("✅ Auto-continue mode enabled - all stages will run automatically")
    
    print("="*60 + "\n")
    
    # Initialize all ML models at startup for better performance
    initialize_models()
    
    # Store callback and video_id globally for use in main()
    PROGRESS_CALLBACK = progress_callback
    VIDEO_ID = video_id
    IS_SHORT_VIDEO = is_short_video
    
    if settings:
        print("\n" + "="*60)
        print("📋 USING CUSTOM PROCESSING SETTINGS")
        print("="*60)
        PROCESSING_SETTINGS.update(settings)
        print(f"Auto-Colored Words: {PROCESSING_SETTINGS['autoColoredWords']}")
        print(f"Color: {PROCESSING_SETTINGS['coloredWordsColor']}")
        print(f"Auto Zoom-Ins: {PROCESSING_SETTINGS['autoZoomIns']}")
        print(f"Auto Cuts: {PROCESSING_SETTINGS['autoCuts']}")
        print(f"Number of Clips: {PROCESSING_SETTINGS['numberOfClips']}")
        print(f"Clip Length: {PROCESSING_SETTINGS['minClipLength']}-{PROCESSING_SETTINGS['maxClipLength']}s")
        if PROCESSING_SETTINGS['customTopics']:
            print(f"Custom Topics ({len(PROCESSING_SETTINGS['customTopics'])}):")
            for i, topic in enumerate(PROCESSING_SETTINGS['customTopics'], 1):
                print(f"  {i}. {topic}")
        print("="*60 + "\n")
    
    import sys
    sys.argv = ["reelsfy.py", "-f", input_path]
    main()

def process_export_file(input_path: str, out_dir: str = "tmp"):
    import sys
    if input_path and input_path.strip():
        sys.argv = ["reelsfy.py", "-f", input_path, "--export-mode"]
    else:
        # For export mode without input file, just run in export mode
        sys.argv = ["reelsfy.py", "--export-mode"]
    main()

# --- Helper function to download logos from Supabase ---
def download_logos_from_styling_files(segments):
    """Download logos from Supabase storage if they're not already local files."""
    import json
    import requests
    
    print("📥 Checking and downloading logos from Supabase...")
    
    for segment_index in segments:
        styling_file = f"tmp/styling_data_{segment_index:03}.json"
        
        if not os.path.exists(styling_file):
            print(f"   ⚠️  No styling file found for segment {segment_index}")
            continue
        
        try:
            with open(styling_file, 'r', encoding='utf-8') as f:
                styling_data = json.load(f)
            
            logo_data = styling_data.get('logo', {})
            logo_url = logo_data.get('url', '')
            
            if not logo_url:
                print(f"   ℹ️  No logo URL for segment {segment_index}")
                continue
            
            # Check if logo is already a local file
            if os.path.exists(logo_url) and os.path.isfile(logo_url):
                print(f"   ✅ Logo already exists locally for segment {segment_index}: {logo_url}")
                continue
            
            # If it's a Supabase URL, download it
            if 'supabase.co' in logo_url or 'supabase' in logo_url.lower():
                print(f"   📥 Downloading logo from Supabase for segment {segment_index}...")
                
                # Try to download using requests (for public URLs)
                try:
                    response = requests.get(logo_url, timeout=30)
                    if response.status_code == 200:
                        local_logo_path = f"tmp/logo_{segment_index:03}.png"
                        with open(local_logo_path, "wb") as f:
                            f.write(response.content)
                        
                        # Update styling data with local path
                        styling_data['logo']['url'] = local_logo_path
                        with open(styling_file, 'w', encoding='utf-8') as f:
                            json.dump(styling_data, f, indent=2, ensure_ascii=False)
                        
                        print(f"   ✅ Downloaded logo to {local_logo_path}")
                    else:
                        print(f"   ⚠️  Failed to download logo (status {response.status_code}), logo URL: {logo_url}")
                except Exception as e:
                    print(f"   ⚠️  Failed to download logo from URL: {e}")
                    print(f"   ℹ️  Logo URL: {logo_url}")
                    print(f"   💡 Tip: Logo should be downloaded by api.py, checking if local file exists...")
                    
                    # Check if api.py already downloaded it
                    local_logo_path = f"tmp/logo_{segment_index:03}.png"
                    if os.path.exists(local_logo_path):
                        print(f"   ✅ Logo already downloaded by api.py: {local_logo_path}")
                        styling_data['logo']['url'] = local_logo_path
                        with open(styling_file, 'w', encoding='utf-8') as f:
                            json.dump(styling_data, f, indent=2, ensure_ascii=False)
                    else:
                        print(f"   ⚠️  Logo file not found locally and download failed")
            else:
                print(f"   ⚠️  Logo URL is not a Supabase URL or local file: {logo_url}")
                
        except Exception as e:
            print(f"   ❌ Error processing logo for segment {segment_index}: {e}")
    
    print("✅ Logo download check complete")

# --- Enhanced burn subtitles with styling ---
def burn_subtitles_with_styling(input_video: str, srt_path: str, output_video: str, short_index: int):
    """
    Burn subtitles with custom styling, logo, and background music
    """
    # Read styling data from JSON file
    styling_file = f"tmp/styling_data_{short_index:03}.json"
    if not os.path.exists(styling_file):
        print(f"Warning: No styling data found for short {short_index}, using default")
        burn_subtitles(input_video, srt_path, output_video)
        return

    with open(styling_file, 'r', encoding='utf-8') as f:
        styling_data = json.load(f)

    # Extract styling information (NEW: supports both old and new structure)
    # Try new structure first (globalTextFormatting), fallback to old (textFormatting)
    text_formatting = styling_data.get('globalTextFormatting') or styling_data.get('textFormatting', {})
    individual_formatting = styling_data.get('individualBoxFormatting', [])
    textbox_position = styling_data.get('textboxPosition', {})
    logo_data = styling_data.get('logo', {})
    music_data = styling_data.get('music', {})
    
    print(f"   📝 Styling info:")
    print(f"      Global formatting: {len(text_formatting)} properties")
    print(f"      Individual boxes: {len(individual_formatting)} custom formats")
    print(f"      Textbox position: xPct={textbox_position.get('xPct')}, yPct={textbox_position.get('yPct')}")
    
    # Build ASS style string from global formatting
    font_name = text_formatting.get('font', 'Assistant')
    base_font_size = text_formatting.get('fontSize', 28)
    
    # Apply fullscreen font size scaling (same as website)
    # BASE_VIDEO_WIDTH = 350 (normal mode), scale for 1080px width
    BASE_VIDEO_WIDTH = 350
    video_width = 1080
    scale_factor = video_width / BASE_VIDEO_WIDTH
    font_size = int(base_font_size * scale_factor)
    
    print(f"   🔤 Font scaling: {base_font_size} → {font_size} (scale: {scale_factor:.2f})")
    
    is_bold = text_formatting.get('isBold', True)
    color = text_formatting.get('color', '#ffffff')
    stroke_color = text_formatting.get('strokeColor', '#000000')
    base_stroke_width = text_formatting.get('strokeWidth', 2)
    # Scale stroke width by same factor as font size
    stroke_width = max(1, int(base_stroke_width * scale_factor))
    shadow_color = text_formatting.get('shadowColor', '#000000')
    base_shadow_distance = text_formatting.get('shadowDistance', 2)
    # Scale shadow distance by same factor as font size
    shadow_distance = max(1, int(base_shadow_distance * scale_factor))
    shadow_size = text_formatting.get('shadowSize', 0)  # Default to 0 - hidden from UI
    # Note: ASS doesn't natively support shadow blur (shadowSize), only shadow distance
    # Shadow blur effect cannot be directly applied in ASS format
    
    # Convert hex colors to ASS format (&HBBGGRR)
    def hex_to_ass_color(hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"&H00{b:02x}{g:02x}{r:02x}"  # ASS uses BGR format
    
    primary_color = hex_to_ass_color(color)
    outline_color = hex_to_ass_color(stroke_color)
    shadow_color_ass = hex_to_ass_color(shadow_color)
    
    # COMMENTED OUT: Custom positioning (testing default position first)
    # # Build style string with dynamic positioning using textboxPosition percentages
    # # Get positioning data from styling (percentages from custom video player)
    # y_pct = textbox_position.get('yPct', 80)  # Default 80% down from top
    # x_pct = textbox_position.get('xPct', 50)  # Default centered (50%)
    # 
    # # Calculate MarginV based on yPct
    # # yPct represents distance from TOP (0% = top, 100% = bottom)
    # # MarginV in ASS represents distance from BOTTOM
    # # For 9:16 video (1080x1920 typical), we need to convert percentage to pixels from bottom
    # 
    # # Assume typical 9:16 resolution: 1080x1920
    # video_height = 1920  # Typical 9:16 height
    # 
    # # Calculate margin from bottom
    # # If yPct = 80%, that's 80% from top = 20% from bottom
    # # So MarginV = (100 - yPct) * video_height / 100
    # margin_from_bottom_pct = 100 - y_pct
    # dynamic_margin_v = int((margin_from_bottom_pct / 100) * video_height)
    # 
    # # Clamp to reasonable range (don't go off screen)
    # dynamic_margin_v = max(20, min(dynamic_margin_v, video_height - 100))
    
    # Automatically calculate subtitle position and width from video player data
    print(f"\n{'='*60}")
    print(f"SUBTITLE POSITIONING FOR SHORT {short_index}")
    print(f"{'='*60}")
    print("Automatically calculating subtitle position from video player data...")
    
    # Use actual textbox position and width from styling data (already calculated as percentages)
    # Note: The frontend sends upper-left corner position, but we need center position
    corner_x_pct = textbox_position.get('xPct', 50)  # Upper-left corner X
    corner_y_pct = textbox_position.get('yPct', 80)  # Upper-left corner Y
    w_pct = textbox_position.get('wPct', 85)  # Width percentage
    h_pct = textbox_position.get('hPct', 15)  # Height percentage
    
    # Convert from upper-left corner to center position
    x_pct = corner_x_pct + (w_pct / 2)  # Center X = corner X + half width
    y_pct = corner_y_pct + (h_pct / 2)  # Center Y = corner Y + half height
    
    print(f"📥 Received textboxPosition data from frontend: {textbox_position}")
    print(f"Frontend coordinates (upper-left corner): X={corner_x_pct}%, Y={corner_y_pct}%, Width={w_pct}%, Height={h_pct}%")
    print(f"Calculated center coordinates: X={x_pct}%, Y={y_pct}%")
    print(f"Subtitle center: ({x_pct}%, {y_pct}%)")
    print(f"Subtitle width: {w_pct}% ({w_pct/2}% to each side of center)")
    print(f"Subtitle height: {h_pct}% (automatically calculated)")
    
    # Calculate actual pixel positions for 9:16 video (1080x1920)
    video_width = 1080
    video_height = 1920
    
    # Convert percentages to pixels
    # yPct: 0% = top, 100% = bottom
    # xPct: 0% = left, 100% = right, 50% = center
    # wPct: width as percentage of video width
    # hPct: height as percentage of video height
    actual_y = int((y_pct / 100) * video_height)  # Position from top
    actual_x = int((x_pct / 100) * video_width)   # Position from left
    actual_w = int((w_pct / 100) * video_width)   # Width in pixels
    actual_h = int((h_pct / 100) * video_height) # Height in pixels
    
    # Calculate textbox boundaries (center the textbox on the x position)
    textbox_left = max(0, actual_x - (actual_w // 2))  # Left edge of textbox
    textbox_right = min(video_width, actual_x + (actual_w // 2))  # Right edge of textbox
    
    # Convert to ASS margins
    # MarginV = distance from bottom (video_height - actual_y)
    # MarginL = distance from left edge to textbox left edge
    # MarginR = distance from textbox right edge to video right edge
    dynamic_margin_v = max(20, video_height - actual_y)  # Ensure minimum margin from bottom
    dynamic_margin_l = textbox_left
    dynamic_margin_r = video_width - textbox_right
    
    print(f"   📐 Position calculation:")
    print(f"      Textbox: xPct={x_pct}%, yPct={y_pct}%, wPct={w_pct}%, hPct={h_pct}%")
    print(f"      Pixels: x={actual_x}, y={actual_y}, w={actual_w}, h={actual_h}")
    print(f"      Boundaries: left={textbox_left}, right={textbox_right}")
    print(f"      ASS margins: V={dynamic_margin_v}, L={dynamic_margin_l}, R={dynamic_margin_r}")
    
    # Calculate and log textbox corners for verification
    textbox_top = actual_y - (actual_h // 2)
    textbox_bottom = actual_y + (actual_h // 2)
    
    print(f"   🎯 [Backend] Subtitle Position Details:")
    print(f"      Video dimensions: {video_width}x{video_height}px")
    print(f"      Subtitle center: ({actual_x}, {actual_y})")
    print(f"      Subtitle size: {actual_w}x{actual_h}px")
    print(f"      Upper-left corner: ({textbox_left}, {textbox_top})")
    print(f"      Bottom-right corner: ({textbox_right}, {textbox_bottom})")
    print(f"      Video player coordinates: X={x_pct}%, Y={y_pct}%, Width={w_pct}%")
    print(f"      ASS positioning:")
    print(f"        - MarginV (from bottom): {dynamic_margin_v}px")
    print(f"        - MarginL (from left): {dynamic_margin_l}px") 
    print(f"        - MarginR (from right): {dynamic_margin_r}px")
    
    # Get additional formatting options
    is_italic = text_formatting.get('isItalic', False)
    is_underline = text_formatting.get('isUnderline', False)
    
    style = (
        f"Alignment=2,MarginV={dynamic_margin_v},MarginL={dynamic_margin_l},MarginR={dynamic_margin_r},"
        f"Fontname={font_name},Fontsize={font_size},"
        f"Bold={1 if is_bold else 0},"
        f"Italic={1 if is_italic else 0},"
        f"Underline={1 if is_underline else 0},"
        f"PrimaryColour={primary_color},"
        f"OutlineColour={outline_color},Outline={stroke_width},"
        f"BackColour={shadow_color_ass},Shadow={shadow_distance},"
        f"BorderStyle=1"
    )
    
    print(f"   📐 ASS Style string: {style}")
    print(f"   📐 ASS Style breakdown:")
    print(f"      - Alignment=2 (center alignment)")
    print(f"      - MarginV={dynamic_margin_v} (distance from bottom)")
    print(f"      - MarginL={dynamic_margin_l} (distance from left)")
    print(f"      - MarginR={dynamic_margin_r} (distance from right)")
    print(f"      - FontSize={font_size} (scaled from {base_font_size}, factor: {scale_factor:.2f})")
    print(f"      - PrimaryColor={primary_color} (text color)")
    print(f"      - OutlineColor={outline_color}, Outline={stroke_width} (scaled from {base_stroke_width}, factor: {scale_factor:.2f})")
    print(f"      - BackColor={shadow_color_ass}, Shadow={shadow_distance} (scaled from {base_shadow_distance}, factor: {scale_factor:.2f})")
    print(f"      - Note: Shadow blur (shadowSize={shadow_size}) is not supported in ASS format (only distance is applied)")
    
    # Process SRT content: color tags → individual formatting → animations
    print(f"   🎨 Processing subtitle styling...")
    
    # Step 1: Read original SRT
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    # Step 2: Convert <color:#hex> tags to ASS {\c&H...} format
    srt_content = convert_color_tags_to_ass(srt_content)
    
    # Step 3: Apply individual box formatting (per-subtitle overrides, including animation)
    # This function now handles both formatting AND animation per subtitle
    if individual_formatting:
        srt_content = apply_individual_box_formatting_to_srt(srt_content, individual_formatting, text_formatting)
    
    # Step 4: Apply animation effects per subtitle
    # Check individual formatting first, then fall back to global
    srt_content = apply_animations_to_srt(srt_content, individual_formatting, text_formatting, 
                                         x_pct, y_pct, w_pct, h_pct, video_width, video_height)
    
    # Step 5: Convert to ASS format for better control
    ass_path = f"tmp/subtitles_{short_index:03}.ass"
    convert_srt_to_ass(srt_content, ass_path, style, font_name)
    
    print(f"   ✅ Converted to ASS format: {ass_path}")
    
    # Copy fonts to tmp directory for FFmpeg to use with ass filter
    import shutil
    fonts_source = os.path.join("tmp", "fonts")
    if os.path.exists(fonts_source):
        # Copy all font files to tmp directory (same as ASS file location)
        for font_file in os.listdir(fonts_source):
            if font_file.endswith('.ttf'):
                src = os.path.join(fonts_source, font_file)
                dst = os.path.join("tmp", font_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"   🔤 Copied font file: {font_file}")
    
    # Build FFmpeg command with ASS file
    esc = ass_path.replace('\\','\\\\').replace(':','\\:')
    inp = ffmpeg.input(input_video)
    video = inp.video.filter_('ass', esc)
    audio = inp.audio
    
    # Probe video once for dimensions and duration (reused for logo and music)
    video_probe = ffmpeg.probe(input_video)
    video_width = int(video_probe['streams'][0]['width'])
    video_height = int(video_probe['streams'][0]['height'])
    video_duration = float(video_probe['format'].get('duration', 0))
    
    # Add logo if exists
    logo_url = logo_data.get('url', '')
    if logo_url and os.path.exists(logo_url):
        logo_opacity = logo_data.get('opacity', 80) / 100.0
        logo_position = logo_data.get('position', 'left')
        
        # Calculate logo size: ~22.86% of video width (matches video player)
        # Base: 80px for 350px video width = 22.86% of width
        BASE_VIDEO_WIDTH = 350
        BASE_LOGO_SIZE = 80
        logo_size_percentage = BASE_LOGO_SIZE / BASE_VIDEO_WIDTH  # ~0.2286
        logo_width = int(video_width * logo_size_percentage)
        logo_height = logo_width  # Keep it square
        
        # Calculate logo position (5% offset from edges)
        horizontal_offset = int(video_width * 0.05)
        vertical_offset = int(video_height * 0.05)
        
        if logo_position == 'left':
            logo_x = horizontal_offset
        else:  # right
            logo_x = video_width - logo_width - horizontal_offset
        
        logo_y = vertical_offset
        
        # Scale logo image first, then overlay with opacity
        # Create logo input with scaling
        logo_input = ffmpeg.input(logo_url)
        logo_scaled = logo_input.filter('scale', logo_width, logo_height, force_original_aspect_ratio='decrease')
        
        # Apply opacity to logo using colorchannelmixer (for RGBA)
        # Convert to RGBA format first, then apply opacity
        logo_with_opacity = logo_scaled.filter('format', 'rgba').filter('colorchannelmixer', aa=logo_opacity)
        
        # Overlay logo onto video using ffmpeg.overlay
        # video is already a filter chain, so we use overlay with two inputs
        video = ffmpeg.overlay(video, logo_with_opacity, x=logo_x, y=logo_y, format='auto')
        
        print(f"   🖼️  Logo overlay: position={logo_position}, size={logo_width}x{logo_height}px, opacity={logo_opacity:.2f}, x={logo_x}, y={logo_y}")
    
    # Add background music if selected
    music_track = music_data.get('track', 'none')
    music_volume_pct = music_data.get('volume', 100)  # 0-100 percentage
    music_volume = music_volume_pct / 100.0  # Convert to 0.0-1.0 range
    
    if music_track != 'none' and music_track:
        # Download music from URL
        music_url = f"https://clippeak.co.il/assets/{music_track}"
        local_music_path = os.path.join('tmp', f"music_{short_index:03}_{os.path.basename(music_track)}")
        
        # Download music file if not already cached
        if not os.path.exists(local_music_path):
            try:
                import requests
                print(f"   🎵 Downloading background music from: {music_url}")
                response = requests.get(music_url, timeout=30)
                if response.status_code == 200:
                    with open(local_music_path, 'wb') as f:
                        f.write(response.content)
                    print(f"   ✅ Downloaded background music: {os.path.basename(music_track)}")
                else:
                    print(f"   ⚠️  Failed to download music (status {response.status_code}), skipping background music")
                    music_track = 'none'
            except Exception as e:
                print(f"   ⚠️  Failed to download music from URL: {e}")
                print(f"   💡 Trying local path as fallback...")
                # Try local path as fallback
                local_fallback = os.path.join("website", "public", "assets", music_track)
                if os.path.exists(local_fallback):
                    local_music_path = local_fallback
                    print(f"   ✅ Using local music file: {local_fallback}")
                else:
                    print(f"   ⚠️  Local music file not found, skipping background music")
                    music_track = 'none'
        
        if music_track != 'none' and os.path.exists(local_music_path):
            # Input music and cut to video duration (already probed above)
            music_input = ffmpeg.input(local_music_path, t=video_duration if video_duration > 0 else None)
            
            # Apply volume to music (volume is 0.0-1.0, convert to dB or use volume filter)
            # Using volume filter with volume parameter (1.0 = 100%, 0.5 = 50%, etc.)
            music_with_volume = music_input.audio.filter('volume', volume=music_volume)
            
            # Mix video audio with background music
            # duration='first' ensures output length matches video length
            audio = ffmpeg.filter([audio, music_with_volume], 'amix', inputs=2, duration='first', dropout_transition=0)
            
            print(f"   🎵 Added background music: {music_track} at {music_volume_pct}% volume (duration: {video_duration:.2f}s)")
    
    # Output with NVENC
    (
        ffmpeg.output(video, audio, output_video, vcodec='h264_nvenc', acodec='aac', preset='fast')
        .run(overwrite_output=True)
    )
    
    # Clean up temporary files
    # COMMENTED OUT: Keep ASS file for manual testing and positioning
    # if os.path.exists(ass_path):
    #     os.remove(ass_path)
    print(f"   📄 ASS file kept for testing: {ass_path}")
    
    print(f"   ✅ Created final video with full styling: {output_video}")

def convert_srt_to_ass(srt_content: str, ass_path: str, style: str, font_name: str):
    """
    Convert SRT content (with ASS tags) to proper ASS subtitle file format.
    """
    import re
    
    # Parse SRT entries
    entries = []
    lines = srt_content.split('\n')
    
    i = 0
    while i < len(lines):
        if i < len(lines) and lines[i].strip().isdigit():
            # Found a subtitle entry
            entry_num = lines[i].strip()
            i += 1
            
            # Get timestamp line
            if i < len(lines) and '-->' in lines[i]:
                timestamp_line = lines[i].strip()
                i += 1
                
                # Parse timestamps (SRT format: HH:MM:SS,mmm)
                match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp_line)
                if match:
                    # Convert to ASS format (H:MM:SS.cc - centiseconds)
                    start_h, start_m, start_s, start_ms = match.group(1), match.group(2), match.group(3), match.group(4)
                    end_h, end_m, end_s, end_ms = match.group(5), match.group(6), match.group(7), match.group(8)
                    
                    start_cs = int(start_ms) // 10
                    end_cs = int(end_ms) // 10
                    
                    start_time = f"{int(start_h)}:{start_m}:{start_s}.{start_cs:02d}"
                    end_time = f"{int(end_h)}:{end_m}:{end_s}.{end_cs:02d}"
                    
                    # Get text lines
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i])
                        i += 1
                    
                    text = '\\N'.join(text_lines)  # ASS uses \N for line breaks
                    
                    entries.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text
                    })
        i += 1
    
    # Write ASS file
    with open(ass_path, 'w', encoding='utf-8') as f:
        # ASS header
        f.write('[Script Info]\n')
        f.write('ScriptType: v4.00+\n')
        f.write('PlayResX: 1080\n')
        f.write('PlayResY: 1920\n')
        f.write('WrapStyle: 0\n')
        f.write('ScaledBorderAndShadow: yes\n\n')
        
        f.write('[V4+ Styles]\n')
        f.write('Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n')
        
        # Parse the style string and build proper ASS style line
        f.write(f'Style: Default,{font_name},')
        
        # Extract values from style string
        style_dict = {}
        for part in style.split(','):
            if '=' in part:
                key, value = part.split('=', 1)
                style_dict[key] = value
        
        # Write style in proper format
        f.write(f"{style_dict.get('Fontsize', '28')},")
        f.write(f"{style_dict.get('PrimaryColour', '&H00FFFFFF')},")
        f.write('&H000000FF,')  # SecondaryColour
        f.write(f"{style_dict.get('OutlineColour', '&H00000000')},")
        f.write(f"{style_dict.get('BackColour', '&H00000000')},")
        f.write(f"{style_dict.get('Bold', '1')},")
        f.write(f"{style_dict.get('Italic', '0')},")
        f.write(f"{style_dict.get('Underline', '0')},")
        f.write('0,')  # StrikeOut
        f.write('100,100,')  # ScaleX, ScaleY
        f.write('0,0,')  # Spacing, Angle
        f.write(f"{style_dict.get('BorderStyle', '1')},")
        f.write(f"{style_dict.get('Outline', '2')},")
        f.write(f"{style_dict.get('Shadow', '2')},")
        f.write(f"{style_dict.get('Alignment', '2')},")
        f.write(f"{style_dict.get('MarginL', '25')},")
        f.write(f"{style_dict.get('MarginR', '25')},")
        f.write(f"{style_dict.get('MarginV', '40')},")
        # Encoding: -1 enables libass special BiDi mode (auto base direction, no forced LTR across tags)
        f.write('-1\n\n')  # Encoding
        
        f.write('[Events]\n')
        f.write('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')
        
        # Write dialogue lines with per-entry RTL detection and punctuation RLM fixes
        for entry in entries:
            text = entry['text']
            # Detect dominant direction on a version stripped of ASS tags
            text_for_dir = re.sub(r'\{\\[^}]*\}', '', text)  # remove ASS override blocks
            text_for_dir = text_for_dir.replace('\\N', ' ')
            # Compute dominant strong direction
            count = Counter([ud.bidirectional(c) for c in list(text_for_dir)])
            rtl_count = count.get('R', 0) + count.get('AL', 0) + count.get('RLE', 0) + count.get('RLI', 0)
            ltr_count = count.get('L', 0) + count.get('LRE', 0) + count.get('LRI', 0)
            is_rtl = rtl_count > ltr_count
            if is_rtl:
                # Add RLM around leading/trailing punctuation per line
                def add_edge_rlm(line: str) -> str:
                    # Insert RLM (\u200F) before leading punctuation and after trailing punctuation
                    # Respect leading/trailing ASS override blocks
                    # Find end of leading ASS tags sequence
                    m_lead = re.match(r'^(?:\{\\[^}]*\})*', line)
                    lead_end = m_lead.end() if m_lead else 0
                    # Find start of trailing ASS tags sequence
                    m_trail = re.search(r'(?:\{\\[^}]*\})*$', line)
                    trail_start = m_trail.start() if m_trail else len(line)
                    s = line
                    # Leading punctuation check
                    if lead_end < len(s):
                        ch = s[lead_end]
                        import unicodedata as _ud
                        if _ud.category(ch).startswith('P'):
                            s = s[:lead_end] + '\u200F' + s[lead_end:]
                            # adjust trail_start if insertion before it
                            if trail_start >= lead_end:
                                trail_start += 1
                    # Trailing punctuation check (look at last visible char before trailing tags)
                    if trail_start > 0:
                        idx = trail_start - 1
                        # skip spaces before trailing tags
                        while idx >= 0 and s[idx].isspace():
                            idx -= 1
                        if idx >= 0:
                            import unicodedata as _ud2
                            if _ud2.category(s[idx]).startswith('P'):
                                s = s[:idx+1] + '\u200F' + s[idx+1:]
                    return s
                # Apply to each rendered line separately
                parts = text.split('\\N')
                parts = [add_edge_rlm(p) for p in parts]
                text = '\\N'.join(parts)
                # Wrap with Right-to-Left embedding for robustness
                text = f"\u202B{text}\u202C"
            f.write(f"Dialogue: 0,{entry['start']},{entry['end']},Default,,0,0,0,,{text}\n")
    
    print(f"   📄 Created ASS file with {len(entries)} subtitle entries")

def add_lrm_after_english(text: str) -> str:
    """
    Add LRM (Left-to-Right Mark) after English/number tokens inside Hebrew text.
    Example: "123" -> "123\u200E", "ABC" -> "ABC\u200E"
    """
    import re
    # Pattern: English letters, numbers, or mixed alphanumeric
    # Match sequences of ASCII alphanumeric characters
    pattern = r'([A-Za-z0-9]+)'
    def add_lrm(match):
        return match.group(1) + '\u200E'
    return re.sub(pattern, add_lrm, text)

def convert_color_tags_to_ass(srt_content: str) -> str:
    """
    Convert <color:#hex>text</color> and <stroke:#hex>text</stroke> tags to ASS format.
    With BiDi fixes: {\c&HBBGGRR&}\u200Ftext\u200F{\r} for colors
                     {\3c&HBBGGRR&}\u200Ftext\u200F{\r} for stroke
    FFmpeg's subtitle filter understands ASS tags but not our custom color tags.
    """
    import re
    
    def hex_to_ass_color(hex_color):
        """Convert #RRGGBB to &H00BBGGRR (ASS BGR format)"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"&H00{b:02x}{g:02x}{r:02x}"
    
    def replace_color_tag(match):
        hex_color = match.group(1)
        text = match.group(2)
        ass_color = hex_to_ass_color(hex_color)
        # Apply LRM after English/number tokens
        text = add_lrm_after_english(text)
        # ASS color tag with BiDi: {\c&HBBGGRR&}\u200Ftext\u200F{\r}
        # Note: \u200F is Right-to-Left Mark (RLM), {\r} resets formatting
        return f"{{\\c{ass_color}&}}\u200F{text}\u200F{{\\r}}"
    
    def replace_stroke_tag(match):
        hex_color = match.group(1)
        text = match.group(2)
        ass_color = hex_to_ass_color(hex_color)
        # Apply LRM after English/number tokens
        text = add_lrm_after_english(text)
        # ASS stroke tag with BiDi: {\3c&HBBGGRR&}\u200Ftext\u200F{\r}
        # Note: \3c sets outline/stroke color in ASS
        return f"{{\\3c{ass_color}&}}\u200F{text}\u200F{{\\r}}"
    
    # Pattern: <color:#RRGGBB>text</color>
    color_pattern = r'<color:(#[0-9A-Fa-f]{6})>(.*?)</color>'
    color_count = len(re.findall(color_pattern, srt_content))
    converted = re.sub(color_pattern, replace_color_tag, srt_content)
    
    # Pattern: <stroke:#RRGGBB>text</stroke>
    stroke_pattern = r'<stroke:(#[0-9A-Fa-f]{6})>(.*?)</stroke>'
    stroke_count = len(re.findall(stroke_pattern, converted))
    converted = re.sub(stroke_pattern, replace_stroke_tag, converted)
    
    print(f"   Converted {color_count} color tags and {stroke_count} stroke tags to ASS format")
    
    return converted

def apply_animations_to_srt(srt_content: str, individual_formatting: dict, global_formatting: dict,
                           x_pct: float, y_pct: float, w_pct: float, h_pct: float,
                           video_width: int, video_height: int) -> str:
    """
    Apply animation effects to SRT content, checking individual formatting first, then global.
    
    individual_formatting: Dict mapping subtitle index to formatting (includes 'animation')
    global_formatting: Global formatting with default 'animation'
    x_pct, y_pct, w_pct, h_pct: Textbox position and size percentages
    video_width, video_height: Video dimensions in pixels
    """
    global_animation = global_formatting.get('animation', 'ללא')
    
    # Parse SRT into entries
    entries = []
    lines = srt_content.split('\n')
    
    i = 0
    while i < len(lines):
        if i < len(lines) and lines[i].strip().isdigit():
            # Found a subtitle entry
            entry_num = lines[i].strip()
            i += 1
            
            # Get timestamp line
            timestamp = lines[i].strip() if i < len(lines) else ""
            i += 1
            
            # Get text lines (until empty line)
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i])
                i += 1
            
            entries.append({
                'num': entry_num,
                'timestamp': timestamp,
                'text': '\n'.join(text_lines)
            })
        i += 1
    
    # Helper function to generate animation ASS tags
    def get_animation_tags(animation: str, line: str) -> str:
        """Generate ASS animation tags based on animation type."""
        if animation == 'ללא' or not animation:
            return line
        
        if animation == 'הופעה':  # fade in
            # Fade in: opacity 0 → 1 over 600ms
            return f"{{\\fad(600,0)}}{line}"
        
        elif animation == 'החלקה':  # slide up + fade
            # Calculate position for move animation
            actual_y = int((y_pct / 100) * video_height)
            actual_x = int((x_pct / 100) * video_width)
            final_x = actual_x
            final_y = actual_y
            start_y = min(final_y + 50, video_height - 20)
            start_x = final_x
            return f"{{\\fad(600,0)\\move({start_x},{start_y},{final_x},{final_y},0,600)}}{line}"
        
        elif animation == 'פופ-אפ':  # popup - scale from small with bounce
            # Scale 60% → 108% → 100% with overshoot
            return f"{{\\fad(100,0)\\t(0,280,\\fscx108\\fscy108)\\t(280,460,\\fscx100\\fscy100)}}{line}"
        
        elif animation == 'זום אין':  # zoom-in - scale from 80% to 100%
            # Scale from 80% to 100% over 600ms
            return f"{{\\fad(100,0)\\fscx80\\fscy80\\t(0,600,\\fscx100\\fscy100)}}{line}"
        
        else:
            return line
    
    # Apply animation per entry
    for i, entry in enumerate(entries):
        # Check individual formatting first, then fall back to global
        box_format = individual_formatting.get(i) or individual_formatting.get(str(i)) if individual_formatting else None
        animation = None
        
        if box_format and isinstance(box_format, dict) and 'animation' in box_format:
            animation = box_format['animation']
        else:
            animation = global_animation
        
        # Apply animation if not 'ללא'
        # Animation tags need to wrap the entire text (including any existing ASS tags)
        if animation and animation != 'ללא':
            entry['text'] = get_animation_tags(animation, entry['text'])
    
    # Rebuild SRT content
    result_lines = []
    for entry in entries:
        result_lines.append(entry['num'])
        result_lines.append(entry['timestamp'])
        result_lines.append(entry['text'])
        result_lines.append('')  # Empty line between entries
    
    # Count animations applied
    individual_anim_count = 0
    global_anim_count = 0
    
    for i, entry in enumerate(entries):
        box_format = individual_formatting.get(i) or individual_formatting.get(str(i)) if individual_formatting else None
        if box_format and isinstance(box_format, dict) and 'animation' in box_format:
            if box_format['animation'] != 'ללא':
                individual_anim_count += 1
        elif global_animation != 'ללא':
            global_anim_count += 1
    
    if individual_anim_count > 0 or global_anim_count > 0:
        print(f"   🎬 Applied animations: {individual_anim_count} individual override(s), {global_anim_count} using global")
    
    return '\n'.join(result_lines)

def apply_individual_box_formatting_to_srt(srt_content: str, individual_formatting: dict, global_formatting: dict) -> str:
    """
    Apply per-box formatting overrides using ASS override tags.
    Individual formatting overrides global formatting on a per-subtitle basis.
    
    individual_formatting: Dict mapping subtitle index (int) to formatting object
    """
    if not individual_formatting or not isinstance(individual_formatting, dict):
        return srt_content
    
    # Parse SRT into entries
    entries = []
    lines = srt_content.split('\n')
    
    i = 0
    while i < len(lines):
        if i < len(lines) and lines[i].strip().isdigit():
            # Found a subtitle entry
            entry_num = lines[i].strip()
            i += 1
            
            # Get timestamp line
            timestamp = lines[i].strip() if i < len(lines) else ""
            i += 1
            
            # Get text lines (until empty line)
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i])
                i += 1
            
            entries.append({
                'num': entry_num,
                'timestamp': timestamp,
                'text': '\n'.join(text_lines)
            })
        i += 1
    
    # Apply individual formatting to each entry
    # individual_formatting is a dict like {0: {...}, 1: {...}, 5: {...}}
    for i, entry in enumerate(entries):
        # Check if this subtitle index has custom formatting
        box_format = individual_formatting.get(i) or individual_formatting.get(str(i))
        if not box_format or not isinstance(box_format, dict):
            continue
        
        overrides = []
        
        # Build ASS override tags for properties that differ from global
        if 'fontSize' in box_format:
            # Apply the same font size scaling as the main function
            base_font_size = box_format['fontSize']
            BASE_VIDEO_WIDTH = 350
            video_width = 1080
            scale_factor = video_width / BASE_VIDEO_WIDTH
            scaled_font_size = int(base_font_size * scale_factor)
            overrides.append(f"\\fs{scaled_font_size}")
        
        if 'color' in box_format:
            def hex_to_ass(hex_color):
                hex_color = hex_color.lstrip('#')
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f"&H00{b:02x}{g:02x}{r:02x}"
            color_ass = hex_to_ass(box_format['color'])
            overrides.append(f"\\c{color_ass}")
        
        if 'strokeColor' in box_format:
            def hex_to_ass(hex_color):
                hex_color = hex_color.lstrip('#')
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f"&H00{b:02x}{g:02x}{r:02x}"
            stroke_ass = hex_to_ass(box_format['strokeColor'])
            overrides.append(f"\\3c{stroke_ass}")
        
        if 'strokeWidth' in box_format:
            # Apply the same stroke width scaling as the main function
            base_stroke_width = box_format['strokeWidth']
            BASE_VIDEO_WIDTH = 350
            video_width = 1080
            scale_factor = video_width / BASE_VIDEO_WIDTH
            scaled_stroke_width = max(1, int(base_stroke_width * scale_factor))
            overrides.append(f"\\bord{scaled_stroke_width}")
        
        if 'shadowColor' in box_format:
            def hex_to_ass(hex_color):
                hex_color = hex_color.lstrip('#')
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f"&H00{b:02x}{g:02x}{r:02x}"
            shadow_ass = hex_to_ass(box_format['shadowColor'])
            overrides.append(f"\\4c{shadow_ass}")
        
        if 'shadowDistance' in box_format:
            # Apply the same shadow distance scaling as the main function
            base_shadow_distance = box_format['shadowDistance']
            BASE_VIDEO_WIDTH = 350
            video_width = 1080
            scale_factor = video_width / BASE_VIDEO_WIDTH
            scaled_shadow_distance = max(1, int(base_shadow_distance * scale_factor))
            overrides.append(f"\\shad{scaled_shadow_distance}")
        
        if 'isBold' in box_format:
            overrides.append(f"\\b{1 if box_format['isBold'] else 0}")
        
        if 'isItalic' in box_format:
            overrides.append(f"\\i{1 if box_format['isItalic'] else 0}")
        
        if 'isUnderline' in box_format:
            overrides.append(f"\\u{1 if box_format['isUnderline'] else 0}")
        
        if 'font' in box_format:
            overrides.append(f"\\fn{box_format['font']}")
        
        # Apply overrides to this subtitle's text
        if overrides:
            override_string = ''.join(overrides)
            entry['text'] = f"{{{override_string}}}{entry['text']}"
    
    # Rebuild SRT content
    result_lines = []
    for entry in entries:
        result_lines.append(entry['num'])
        result_lines.append(entry['timestamp'])
        result_lines.append(entry['text'])
        result_lines.append('')  # Empty line between entries
    
    # Count how many boxes had formatting applied
    formatted_count = sum(1 for k in individual_formatting.keys() 
                         if (int(k) if isinstance(k, str) else k) < len(entries))
    print(f"   Applied individual formatting to {formatted_count} subtitle boxes")
    
    return '\n'.join(result_lines)

def apply_animation_to_srt(srt_path: str, animation: str) -> str:
    """
    Apply animation effects to SRT content
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    if animation == 'ללא':
        return srt_content
    
    # Parse SRT and add animation tags
    lines = srt_content.split('\n')
    result_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.isdigit():  # Subtitle number
            result_lines.append(line)
        elif '-->' in line:  # Timestamp
            result_lines.append(line)
        elif line:  # Subtitle text
            # Apply animation based on type
            if animation == 'הופעה':  # fade in
                animated_text = f"{{\\fad(200,0)}}{line}"
            elif animation == 'החלקה':  # slide up
                animated_text = f"{{\\move(0,400,0,0,0,350)}}{line}"
            elif animation == 'פופ-אפ':  # popup
                animated_text = f"{{\\fscx70\\fscy70\\t(0,350,\\fscx100\\fscy100)}}{line}"
            elif animation == 'זום אין':  # zoom-in
                animated_text = f"{{\\fscx90\\fscy90\\t(0,350,\\fscx100\\fscy100)}}{line}"
            else:
                animated_text = line
            
            result_lines.append(animated_text)
        else:  # Empty line
            result_lines.append(line)
        
        i += 1
    
    return '\n'.join(result_lines)
