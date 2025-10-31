# Python Code Structure Documentation

## Overview
The Python backend consists of three main components:
1. **`api.py`** - FastAPI backend server that handles HTTP requests from the frontend
2. **`reelsfy_folder/reelsfy.py`** - Main video processing engine with all video manipulation logic
3. **`syncnet_repo/`** - External repository for audio-visual synchronization (SyncNet)
4. **`reelsfy_folder/utils/auto-subtitle/`** - Subtitle generation utilities

---

## Directory Structure

```
D:\ClipPeak\
├── api.py                          # FastAPI backend server
├── reelsfy_folder/
│   ├── reelsfy.py                  # Main video processing engine (4048 lines)
│   ├── __init__.py                 # Empty package init
│   ├── requirements.txt            # Python dependencies
│   ├── test_faster_whisper.py      # Test script for Whisper
│   ├── tmp/                        # Temporary processing files
│   │   ├── fonts/                  # Font files for subtitles
│   │   ├── *.mp4                   # Video files during processing
│   │   └── *.srt                   # Subtitle files
│   ├── utils/
│   │   └── auto-subtitle/          # Subtitle generation package
│   │       ├── auto_subtitle/
│   │       │   ├── __init__.py     # Package init
│   │       │   ├── cli.py          # CLI interface for subtitle generation
│   │       │   └── utils.py        # Utility functions for subtitles
│   │       └── setup.py            # Package setup
│   └── results/                    # Output results
└── syncnet_repo/                   # SyncNet repository (external)
    ├── SyncNetInstance.py          # Main SyncNet interface
    ├── SyncNetModel.py             # SyncNet model definition
    ├── detectors/                  # Face detection utilities
    │   └── s3fd/                   # S3FD face detector
    ├── syncnet_v2.model            # Pre-trained model file
    └── requirements.txt            # SyncNet dependencies
```

---

## 1. `api.py` - FastAPI Backend Server

**Location**: `D:\ClipPeak\api.py`

**Purpose**: Main HTTP API server that receives requests from the React frontend and orchestrates video processing.

### Imports
```python
import os, shutil, subprocess, json, time, re
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, BackgroundTasks
from supabase import create_client
from reelsfy_folder.reelsfy import process_video_file, process_export_file
```

### Key Functions

#### `process_video(payload, background_tasks)`
- **Route**: `POST /process-video`
- **Purpose**: Main entry point for video processing requests
- **Parameters**:
  - `payload`: Contains video metadata, settings, export mode flag
  - `background_tasks`: FastAPI background task handler
- **Modes**:
  - **Upload Mode**: New video uploaded from frontend
  - **Export Mode**: Final export with subtitle burning and styling

#### `update_progress(video_id, progress, stage, eta_seconds)`
- **Purpose**: Updates video processing progress in Supabase database
- **Real-time**: Frontend polls/uses WebSocket to get progress updates

#### `run_reelsfy(bucket, file_key, user_email, settings, is_short_video)`
- **Purpose**: Background task for processing uploaded videos
- **Workflow**:
  1. Downloads video from Supabase storage
  2. Calls `process_video_file()` from `reelsfy.py`
  3. Uploads processed videos back to Supabase
  4. Updates database with results

#### `run_export_processing(user_folder_id, video_folder_name, user_email, video_id, export_data)`
- **Purpose**: Background task for final video export
- **Workflow**:
  1. Downloads processed videos from Supabase
  2. Calls `process_export_file()` for each short
  3. Burns subtitles, logos, and background music
  4. Uploads final videos back to Supabase

### Dependencies
- `fastapi`: Web framework
- `supabase`: Database and storage client
- `reelsfy_folder.reelsfy`: Imports main processing functions

---

## 2. `reelsfy_folder/reelsfy.py` - Main Video Processing Engine

**Location**: `D:\ClipPeak\reelsfy_folder\reelsfy.py`  
**Size**: 4048 lines  
**Purpose**: Core video processing logic - transcription, segmentation, face tracking, subtitle generation, etc.

### Imports

#### Standard Library
```python
import os, sys, json, shutil, subprocess, argparse, math, re
from datetime import datetime
import unicodedata as ud
from collections import Counter
```

#### Third-Party Libraries
```python
import cv2                    # OpenCV - video processing, face tracking
import ffmpeg                 # FFmpeg-python - video/audio manipulation
from pytube import YouTube    # YouTube video download
from openai import OpenAI      # OpenAI API for GPT-4 analysis
import mediapipe as mp        # MediaPipe - face detection and mesh
```

#### Optional/On-Demand Imports
- `resemblyzer`: Voice encoder for speaker diarization
- `spectralcluster`: Clustering for speaker separation
- `webrtcvad`: Voice activity detection
- `soundfile`, `librosa`: Audio processing
- `requests`: HTTP requests for downloading music/logos
- `faster_whisper`: Faster Whisper transcription (alternative to Whisper)
- `torch`: PyTorch for SyncNet (loaded conditionally)

### Key Global Variables

```python
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

PROGRESS_CALLBACK = None  # Function to report progress
VIDEO_ID = None          # Current video ID for progress tracking
IS_SHORT_VIDEO = False   # Flag for short video mode (< 3 minutes)
SKIP_MODE = False        # Auto-continue mode (no prompts)

# Global ML model instances (loaded at startup for performance)
GLOBAL_FACE_DETECTION = None
GLOBAL_FACE_MESH = None
GLOBAL_SYNCNET = None
GLOBAL_VOICE_ENCODER = None
```

### Core Functions by Category

#### **A. Model Management**

##### `initialize_models()`
- **Purpose**: Pre-loads all ML models at startup
- **Models**: MediaPipe Face Detection, Face Mesh, SyncNet, Voice Encoder
- **Performance**: Reduces loading time from ~10 minutes to instant for subsequent videos

##### `cleanup_models()`
- **Purpose**: Releases ML model resources when done

#### **B. Transcription & Subtitles**

##### `generate_transcript(input_file) -> (transcript, language)`
- **Purpose**: Generates full video transcript using `faster-whisper` or `auto_subtitle`
- **Returns**: SRT content string and detected language code
- **Fallback**: Uses `auto_subtitle` CLI if faster-whisper fails

##### `generate_subtitle_for_clip(input_path, detected_language) -> srt_path`
- **Purpose**: Generates subtitles for individual video segments
- **Features**: 
  - Enforces max 5 words per subtitle, min 1.0s duration
  - RTL-aware (Hebrew) formatting
  - Uses detected language from main transcript

##### `transcribe_clip_with_faster_whisper(input_path, detected_language) -> srt_path`
- **Purpose**: Fast transcription using faster-whisper library
- **GPU Support**: Uses CUDA if available

#### **C. GPT-4 Analysis**

##### `generate_viral(transcript) -> segments_dict`
- **Purpose**: Analyzes transcript to find viral segments
- **Returns**: Dictionary with segments, titles, descriptions
- **Output Format**:
  ```python
  {
    "segments": [
      {
        "start": "00:01:23",
        "end": "00:02:45",
        "title": "כותרת",
        "description": "תיאור"
      }
    ]
  }
  ```

##### `generate_short_video_styling(transcript, auto_zoom, color_hex) -> styling_dict`
- **Purpose**: For short videos - generates colored words and zoom cues
- **Returns**:
  ```python
  {
    "title": "כותרת #תגית",
    "description": "תיאור #תגית",
    "srt_overrides": {
      "1": "<color:#FF3B3B>מילה</color> <zoom>חשובה</zoom>"
    }
  }
  ```

#### **D. Video Processing**

##### `generate_segments(segments)`
- **Purpose**: Extracts video segments using FFmpeg
- **Output**: `output000.mp4`, `output001.mp4`, etc. in `tmp/`

##### `remove_silence_with_ffmpeg(input_video, output_video, ...)`
- **Purpose**: Removes silent sections from video
- **Method**: Uses FFmpeg's `silencedetect` filter
- **Parameters**: dB threshold (-30dB default), min duration (1.0s)

##### `generate_short(input_file, output_file, srt_path, ...)`
- **Purpose**: Simple face tracking for 9:16 cropping (fallback)
- **Method**: Basic MediaPipe face detection

##### `generate_short_advanced_asd(in_path, out_path, srt_path, ...)`
- **Purpose**: Advanced active speaker detection with multiple speakers
- **Features**:
  - Audio diarization (speaker separation)
  - Face tracking with MediaPipe
  - SyncNet integration (audio-visual sync)
  - Mouth movement detection (fallback)
  - 50% center deadzone for smooth tracking
  - Crops to 9:16 aspect ratio

##### `generate_short_simple_fallback(in_path, out_path, ...)`
- **Purpose**: Basic face tracking fallback if advanced ASD fails

##### `apply_zoom_effects_only(input_file, output_file, srt_path)`
- **Purpose**: Applies zoom effects without face tracking (for 9:16 videos)

##### `generate_face_annotation_video(in_path, out_path, tracks, spk2face, ...)`
- **Purpose**: Creates debug video showing face bounding boxes and confidence scores
- **Output**: Visualizes which speaker is active at each moment

#### **E. Subtitle Processing**

##### `parse_srt(srt_path) -> entries`
- **Purpose**: Parses SRT file into structured format

##### `write_srt_entries(entries, srt_path, rtl_wrap=True)`
- **Purpose**: Writes subtitle entries to SRT file
- **RTL**: Adds Right-to-Left embedding characters for Hebrew

##### `enforce_srt_word_chunks(entries, max_words=5, min_dur=1.0)`
- **Purpose**: Splits long subtitles into chunks (max 5 words, min 1.0s)

##### `apply_srt_overrides(srt_path, srt_overrides)`
- **Purpose**: Applies GPT-4 styling (colored words, zoom tags) to SRT

##### `remove_zoom_tags_from_srt(srt_path)`
- **Purpose**: Removes `<zoom>` tags from final SRT files

##### `convert_srt_to_ass(srt_content, ass_path, style, font_name)`
- **Purpose**: Converts SRT to ASS format for FFmpeg subtitle burning
- **Features**: 
  - RTL embedding (`\u202B...\u202C`)
  - Encoding: -1 (for BiDi support)
  - Font styling

##### `convert_color_tags_to_ass(srt_content)`
- **Purpose**: Converts `<color:#RRGGBB>text</color>` to ASS format
- **BiDi Fix**: Adds RLM (`\u200F`) around colored text for Hebrew

##### `apply_individual_box_formatting_to_srt(srt_content, individual_formatting, global_formatting)`
- **Purpose**: Applies per-subtitle formatting (colors, strokes, shadows)
- **Scaling**: Scales stroke width and shadow distance based on font size

##### `apply_animations_to_srt(srt_content, individual_formatting, global_formatting)`
- **Purpose**: Applies animation effects to subtitles
- **Animations**:
  - `'הופעה'` (Fade In): `\fad` tags
  - `'החלקה'` (Slide Up): `\move` tags
  - `'פופ-אפ'` (Popup): `\t` with scale
  - `'זום אין'` (Zoom In): `\fscx`, `\fscy` with scale

#### **F. Export & Final Processing**

##### `download_logos_from_styling_files(segments)`
- **Purpose**: Downloads logo files from Supabase for each segment

##### `burn_subtitles_with_styling(input_video, srt_path, output_video, short_index)`
- **Purpose**: Final video export with all styling applied
- **Features**:
  - Burns ASS subtitles with formatting
  - Overlays logo (position, opacity, size from JSON)
  - Mixes background music (downloaded from URL, volume from JSON)
  - Trims music to video duration
  - Applies stroke, shadow, color formatting

#### **G. Main Entry Points**

##### `process_video_file(input_path, out_dir, settings, video_id, progress_callback, is_short_video)`
- **Purpose**: Main function for processing uploaded videos
- **Called By**: `api.py` → `run_reelsfy()`
- **Workflow** (Regular Mode):
  1. Download/copy video
  2. Generate transcript
  3. GPT-4 viral segment detection
  4. Extract segments
  5. Remove silence (if enabled)
  6. Generate subtitles
  7. Crop with face tracking
  8. Remove `<zoom>` tags from SRT
  9. Upload to Supabase

- **Workflow** (Short Video Mode):
  1. Download/copy video
  2. Remove silence (if enabled, before transcription)
  3. Generate transcript
  4. GPT-4 styling (colored words, zoom)
  5. Crop to 9:16 (if not already 9:16) or apply zoom only
  6. Upload to Supabase

##### `process_export_file(input_path, out_dir)`
- **Purpose**: Final export processing
- **Called By**: `api.py` → `run_export_processing()`
- **Workflow**:
  1. Download logos from Supabase
  2. Burn subtitles with styling
  3. Overlay logo
  4. Mix background music
  5. Upload final video to Supabase

##### `main()`
- **Purpose**: CLI entry point (for standalone execution)
- **Usage**: `python reelsfy.py --file video.mp4 [options]`

---

## 3. `reelsfy_folder/utils/auto-subtitle/` - Subtitle Utilities

**Location**: `D:\ClipPeak\reelsfy_folder\utils\auto-subtitle\`

### Package Structure
- `auto_subtitle/__init__.py`: Package initialization
- `auto_subtitle/cli.py`: CLI interface for subtitle generation
- `auto_subtitle/utils.py`: Utility functions (timestamp formatting, SRT writing)

### Key Functions (`cli.py`)

##### `main()`
- **Purpose**: CLI entry point for subtitle generation
- **Usage**: `auto_subtitle video.mp4 --srt_only True --model turbo`
- **Features**:
  - Uses OpenAI Whisper or faster-whisper
  - Generates SRT files
  - Can burn subtitles onto video (if `--srt_only` is False)

### Key Functions (`utils.py`)

##### `format_timestamp(seconds, always_include_hours) -> str`
- **Purpose**: Formats timestamp for SRT (HH:MM:SS.mmm)

##### `write_srt(transcript, file)`
- **Purpose**: Writes transcript segments to SRT file

##### `filename(path) -> str`
- **Purpose**: Extracts filename without extension

---

## 4. `syncnet_repo/` - SyncNet Audio-Visual Synchronization

**Location**: `D:\ClipPeak\syncnet_repo\`

**Purpose**: External repository for SyncNet - determines which face is speaking by analyzing audio-visual synchronization.

### Key Files

##### `SyncNetInstance.py`
- **Class**: `SyncNetInstance`
- **Purpose**: Main interface for SyncNet model
- **Method**: `evaluate(opt, videofile)` - Scores audio-visual sync

##### `SyncNetModel.py`
- **Class**: `S` (SyncNet model architecture)
- **Purpose**: PyTorch model definition for SyncNet

##### `detectors/s3fd/`
- **Purpose**: Face detection for SyncNet input
- **Files**: `nets.py`, `box_utils.py`, `__init__.py`

### Usage in `reelsfy.py`
- **Conditional Import**: Only loaded if `SYNCNET_DIR` and `SYNCNET_MODEL` exist
- **Hardcoded Paths**:
  ```python
  SYNCNET_DIR = r"D:\ClipPeak\syncnet_repo"
  SYNCNET_MODEL = r"D:\ClipPeak\syncnet_repo\syncnet_v2.model"
  ```
- **Fallback**: If SyncNet unavailable, uses mouth movement detection

---

## 5. File Dependencies & Data Flow

### Processing Flow

```
Frontend (React)
    ↓ HTTP POST
api.py (FastAPI)
    ↓ Background Task
reelsfy.py::process_video_file()
    ↓
    ├─→ generate_transcript() → faster-whisper / auto_subtitle
    ├─→ generate_viral() → OpenAI GPT-4 API
    ├─→ generate_segments() → FFmpeg
    ├─→ remove_silence_with_ffmpeg() → FFmpeg
    ├─→ generate_subtitle_for_clip() → faster-whisper
    ├─→ generate_short_advanced_asd() → MediaPipe + SyncNet + Resemblyzer
    └─→ Upload to Supabase Storage
```

### Export Flow

```
Frontend (React)
    ↓ HTTP POST (export data)
api.py::run_export_processing()
    ↓
    ├─→ download_logos_from_styling_files() → Supabase Storage
    ├─→ burn_subtitles_with_styling() → FFmpeg
    │   ├─→ convert_srt_to_ass() → ASS conversion
    │   ├─→ apply_individual_box_formatting_to_srt() → Formatting
    │   ├─→ apply_animations_to_srt() → Animations
    │   ├─→ Download music from URL → requests
    │   └─→ FFmpeg overlay (logo, subtitles, music)
    └─→ Upload to Supabase Storage
```

### External Dependencies

- **Supabase**: Database (video metadata) + Storage (video files, logos)
- **OpenAI API**: GPT-4 for viral segment detection and styling
- **FFmpeg**: All video/audio processing
- **MediaPipe**: Face detection and tracking
- **CUDA**: GPU acceleration (optional, for faster-whisper and SyncNet)

---

## 6. Configuration & Environment

### Hardcoded Paths (in `reelsfy.py`)
```python
SYNCNET_DIR = r"D:\ClipPeak\syncnet_repo"
SYNCNET_MODEL = r"D:\ClipPeak\syncnet_repo\syncnet_v2.model"
```

### Environment Variables
- **Supabase**: Credentials in `api.py` (hardcoded)
- **OpenAI API**: Key in `reelsfy.py` (hardcoded)

### Settings (from Frontend)
- Processed via `PROCESSING_SETTINGS` dictionary
- Passed from React frontend through `api.py` → `process_video_file()`

---

## 7. Temporary Files Structure

### `tmp/` Directory
```
tmp/
├── fonts/                    # Font files for subtitle rendering
├── input_video.mp4          # Original uploaded video
├── input_video.srt          # Full transcript
├── input_video_nosilence.mp4 # After silence removal (short videos)
├── output000.mp4            # Extracted segment 0
├── output001.mp4            # Extracted segment 1
├── output_nosilence000.mp4 # Segment 0 after silence removal
├── output_cropped000.mp4    # Final cropped segment 0
├── output_cropped000.srt    # Final SRT for segment 0
└── ...
```

---

## 8. Key Processing Workflows

### Regular Video Mode (< 3 minutes not flagged)
1. Download/copy → `tmp/input_video.mp4`
2. Transcribe → `tmp/input_video.srt`
3. GPT-4 viral analysis → `segments` dict
4. Extract segments → `tmp/output000.mp4`, `output001.mp4`, ...
5. Remove silence (per segment) → `tmp/output_nosilence000.mp4`
6. Generate subtitles (per segment) → `tmp/output_nosilence000.srt`
7. Crop with face tracking → `tmp/output_cropped000.mp4`
8. Remove `<zoom>` tags → `tmp/output_cropped000.srt` (final)
9. Upload to Supabase

### Short Video Mode (< 3 minutes, flagged)
1. Download/copy → `tmp/input_video.mp4`
2. Remove silence (optional) → `tmp/input_video_nosilence.mp4`
3. Transcribe → `tmp/input_video_nosilence.srt` or `input_video.srt`
4. GPT-4 styling → `styling_data.json` (colored words, zoom, title, description)
5. Check aspect ratio:
   - **If 9:16**: Apply zoom only → `tmp/output_cropped000.mp4`
   - **Else**: Crop with face tracking → `tmp/output_cropped000.mp4`
6. Finalize SRT (remove `<zoom>`) → `tmp/output_cropped000.srt`
7. Upload to Supabase

### Export Mode
1. Download processed videos from Supabase
2. Download logos from Supabase (if in styling JSON)
3. For each video:
   - Convert SRT to ASS with formatting
   - Apply individual subtitle formatting
   - Apply animations
   - Burn subtitles with FFmpeg
   - Overlay logo (position, opacity, size)
   - Mix background music (volume, trimmed to video duration)
4. Upload final videos to Supabase

---

## 9. Dependencies Summary

### `reelsfy_folder/requirements.txt`
```
pytube                    # YouTube video download
opencv-python             # Video processing, face tracking
openai                    # GPT-4 API
python-dotenv             # Environment variables
opencv-contrib-python     # Extended OpenCV features
faster-whisper            # Fast transcription
transformers              # HuggingFace models
huggingface-hub           # Model downloads
mediapipe                 # Face detection and mesh
resemblyzer               # Voice encoder for speaker diarization
spectralcluster           # Speaker clustering
webrtcvad                 # Voice activity detection
soundfile                 # Audio file I/O
librosa                   # Audio analysis
requests                  # HTTP requests for music/logo downloads
python_speech_features    # Audio features for SyncNet
```

### `syncnet_repo/requirements.txt`
- PyTorch (with CUDA support)
- OpenCV
- SciPy
- NumPy
- `python_speech_features`

---

## 10. Entry Points

### From Frontend (React)
1. **Upload Video**: `POST /process-video` (payload with video metadata)
2. **Export Videos**: `POST /process-video` (export mode, with styling data)

### Standalone CLI
```bash
python reelsfy_folder/reelsfy.py --file video.mp4 [--short] [--export]
```

### From `api.py`
- `run_reelsfy()` → `process_video_file()`
- `run_export_processing()` → `process_export_file()`

---

## 11. Key Technologies

- **FastAPI**: HTTP API framework
- **FFmpeg**: Video/audio processing (encoding, filtering, burning)
- **OpenCV**: Video frame processing, face tracking
- **MediaPipe**: Face detection, face mesh (lip tracking)
- **PyTorch**: SyncNet, model inference
- **Faster-Whisper**: Speech-to-text transcription
- **Resemblyzer**: Voice encoding for speaker separation
- **OpenAI GPT-4**: AI analysis for viral segments and styling
- **Supabase**: Database and cloud storage

---

This documentation provides a comprehensive overview of the Python codebase structure, dependencies, and data flow. For specific implementation details, refer to the code comments in each file.

