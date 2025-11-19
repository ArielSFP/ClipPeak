import os
import glob
import shutil
import subprocess
import json
import time
import re
import threading
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, BackgroundTasks
from supabase import create_client
from reelsfy_folder.reelsfy import process_video_file, process_export_file, initialize_models  # your existing script, refactored into importable functions

# Load Supabase credentials from environment variables (Cloud Run secrets)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://bzyclxmakfklbxnsradh.supabase.co/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ6eWNseG1ha2ZrbGJ4bnNyYWRoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAwNTE2MTEsImV4cCI6MjA2NTYyNzYxMX0.2iHLGirSBn4__qnJ5gqIbUER1QHafmVHV5UMw4_qGYo")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print(f"🔐 Using Supabase URL: {SUPABASE_URL}")
print(f"🔐 Supabase key loaded: {SUPABASE_KEY[:20]}..." if SUPABASE_KEY else "🔐 No Supabase key found")

app = FastAPI()

# Model loading state (thread-safe)
MODEL_LOADING_LOCK = threading.Lock()
MODELS_LOADING = False
MODELS_LOADED = False
MODEL_LOAD_ERROR = None

def load_models_in_background():
    """Load TalkNet models in a background thread."""
    global MODELS_LOADING, MODELS_LOADED, MODEL_LOAD_ERROR
    
    with MODEL_LOADING_LOCK:
        if MODELS_LOADED or MODELS_LOADING:
            return  # Already loaded or loading
        MODELS_LOADING = True
    
    print("\n" + "="*60)
    print("🚀 BACKGROUND: Loading TalkNet models in background thread...")
    print("="*60)
    
    try:
        initialize_models()
        with MODEL_LOADING_LOCK:
            MODELS_LOADED = True
            MODELS_LOADING = False
            MODEL_LOAD_ERROR = None
        print("✅ TalkNet models loaded and cached in memory")
        print("   Models will be reused for all subsequent requests")
        print("="*60 + "\n")
    except Exception as e:
        import traceback
        traceback.print_exc()
        with MODEL_LOADING_LOCK:
            MODELS_LOADING = False
            MODELS_LOADED = False
            MODEL_LOAD_ERROR = str(e)
        print(f"❌ Failed to load TalkNet models in background: {e}")
        print("   Models will be loaded on first video request instead")
        print("="*60 + "\n")

# Start loading models in background thread immediately (non-blocking)
@app.on_event("startup")
async def startup_event():
    """Start background model loading - FastAPI starts immediately."""
    print("\n" + "="*60)
    print("🚀 FASTAPI STARTUP: Starting background model loading...")
    print("="*60)
    print("   API is ready to accept requests while models load")
    print("="*60 + "\n")
    
    # Start model loading in background thread (non-blocking)
    model_thread = threading.Thread(target=load_models_in_background, daemon=True)
    model_thread.start()
    print("📦 Background thread started for TalkNet model loading")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://192.168.1.32:8080",
        "https://clippeak.co.il",
        "https://www.clippeak.co.il",
         "*",  # (optional for local dev)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint for Cloud Run
@app.get("/")
async def root():
    """Root endpoint - returns service info."""
    return {
        "service": "ClipPeak Video Processing API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancer."""
    try:
        import torch
        from reelsfy_folder.reelsfy import GLOBAL_TALKNET, GLOBAL_TALKNET_DET
        
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
        
        with MODEL_LOADING_LOCK:
            models_loaded = MODELS_LOADED and (GLOBAL_TALKNET is not None and GLOBAL_TALKNET_DET is not None)
            models_loading = MODELS_LOADING
            model_load_error = MODEL_LOAD_ERROR
        
        model_status = "loaded" if models_loaded else ("loading" if models_loading else "not_loaded")
        
        return {
            "status": "healthy",
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
            "models_loaded": models_loaded,
            "models_loading": models_loading,
            "model_status": model_status,
            "model_load_error": model_load_error,
            "supabase_connected": bool(SUPABASE_URL and SUPABASE_KEY),
            "openai_configured": bool(os.environ.get("OPENAI_API_KEY"))
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/process-video")
async def process_video(payload: dict, background_tasks: BackgroundTasks):
    # Check export mode first
    export_mode = payload.get("exportMode", False)
    video_id = payload.get("videoId")
    user_email = payload.get("email")  # optional, for notifications
    settings = payload.get("settings", {})  # Processing settings from popup
    is_short_video = payload.get("isShortVideo", False)  # Explicit short video mode flag

    print(f"Starting video processing (export_mode={export_mode}, short_video={is_short_video}), updating status...")
    
    if settings:
        print(f"📋 Processing Settings:")
        print(f"   Auto-Colored Words: {settings.get('autoColoredWords')}")
        print(f"   Color: {settings.get('coloredWordsColor')}")
        print(f"   Auto Zoom-Ins: {settings.get('autoZoomIns')}")
        print(f"   Auto Cuts: {settings.get('autoCuts')}")
        print(f"   Number of Clips: {settings.get('numberOfClips')}")
        print(f"   Clip Length Range: {settings.get('minClipLength')}-{settings.get('maxClipLength')}s")
        print(f"   Custom Topics: {len(settings.get('customTopics', []))} topics")
    
    if export_mode:
        # For export mode, we're processing existing shorts
        # Export data is sent directly in the request (no storage download needed)
        export_data = payload.get("exportData", {})
        user_folder_id = payload.get("userFolderId", "")
        video_folder_name = payload.get("videoFolderName", "")
        
        print(f"📦 Export mode - Processing {len(export_data.get('shorts', []))} shorts")
        print(f"   User folder: {user_folder_id}")
        print(f"   Video folder: {video_folder_name}")
        
        # Don't update status for export mode - handle it in the processing function
        background_tasks.add_task(
            run_export_processing, 
            user_folder_id, 
            video_folder_name, 
            user_email, 
            video_id, 
            export_data
        )
    else:
        # For upload mode, we're processing new videos
        bucket = payload["bucket"]        # e.g. "videos"
        file_key = payload["fileKey"]     # path in the bucket
        
        print(f"📤 Upload mode - Processing video from storage")
        print(f"   Bucket: {bucket}")
        print(f"   File key: {file_key}")
        
        supabase.table("videos").update({"status": "processing"}).eq("file_url", file_key).execute()
        background_tasks.add_task(run_reelsfy, bucket, file_key, user_email, settings, is_short_video)

    return {"status": "processing" if not export_mode else "completed"}


def update_progress(video_id: str, progress: int, stage: str = None, eta_seconds: int = None):
    """Update video processing progress in database for real-time UI updates."""
    try:
        update_data = {"progress": progress}
        # Note: stage parameter kept for logging but not stored in database
        # Note: eta_seconds not stored either (can be added later if needed)
        
        supabase.table("videos").update(update_data).eq("id", video_id).execute()
        print(f"📊 Progress: {progress}% - {stage if stage else 'Processing...'}")
    except Exception as e:
        print(f"Warning: Could not update progress: {e}")

def run_reelsfy(bucket: str, file_key: str, user_email: str, settings: dict = None, is_short_video: bool = False):
    """
    Run reelsfy processing with custom settings from frontend.
    
    settings: {
        'autoColoredWords': bool,
        'coloredWordsColor': str (hex),
        'autoZoomIns': bool,
        'autoCuts': bool,
        'numberOfClips': int,
        'minClipLength': int,
        'maxClipLength': int,
        'customTopics': list[str]
    }
    is_short_video: Explicit flag for videos under 3 minutes
    """
    # Start timing
    start_time = time.time()
    
    # 0) Ensure models are loaded (should already be loaded in background, but check anyway)
    print("\n" + "="*60)
    print("STAGE: Check TalkNet models")
    print("="*60)
    print("▶️  RUNNING: Check TalkNet models (auto-continue mode)\n")
    
    # Declare global variables to avoid UnboundLocalError (must be before any usage)
    global MODELS_LOADING, MODELS_LOADED, MODEL_LOAD_ERROR
    
    try:
        from reelsfy_folder.reelsfy import GLOBAL_TALKNET, GLOBAL_TALKNET_DET
        
        # Check if models are loaded
        with MODEL_LOADING_LOCK:
            models_ready = MODELS_LOADED and (GLOBAL_TALKNET is not None and GLOBAL_TALKNET_DET is not None)
            still_loading = MODELS_LOADING and not MODELS_LOADED
    except Exception as e:
        print(f"⚠️  Error checking model status: {e}")
        print("   Continuing with model loading attempt...")
        models_ready = False
        still_loading = False
    
    if models_ready:
        print("✅ TalkNet models already loaded (from background thread)\n")
    elif still_loading:
        # Import here to avoid issues if import failed above
        try:
            from reelsfy_folder.reelsfy import GLOBAL_TALKNET, GLOBAL_TALKNET_DET
        except ImportError:
            GLOBAL_TALKNET = None
            GLOBAL_TALKNET_DET = None
        print("⏳ TalkNet models are still loading in background...")
        print("   Waiting for models to finish loading...")
        # Wait for models to load (with timeout)
        max_wait = 300  # 5 minutes max wait
        wait_interval = 2  # Check every 2 seconds
        waited = 0
        while waited < max_wait:
            time.sleep(wait_interval)
            waited += wait_interval
            try:
                with MODEL_LOADING_LOCK:
                    if MODELS_LOADED:
                        # Re-import to get latest model references
                        try:
                            from reelsfy_folder.reelsfy import GLOBAL_TALKNET, GLOBAL_TALKNET_DET
                            if GLOBAL_TALKNET is not None and GLOBAL_TALKNET_DET is not None:
                                print(f"✅ TalkNet models loaded successfully (waited {waited}s for background loading)\n")
                                break
                        except ImportError:
                            pass
                    if not MODELS_LOADING and MODEL_LOAD_ERROR:
                        # Background loading failed, try to load now
                        print(f"⚠️  Background loading failed: {MODEL_LOAD_ERROR}")
                        print("   Attempting to load models now...")
                        try:
                            initialize_models()
                            with MODEL_LOADING_LOCK:
                                MODELS_LOADED = True
                                MODELS_LOADING = False
                                MODEL_LOAD_ERROR = None
                            print("✅ TalkNet models loaded successfully\n")
                            break
                        except Exception as e:
                            print(f"❌ Failed to load TalkNet models: {e}")
                            print("⚠️  Video processing may fail without TalkNet\n")
                            break
            except Exception as e:
                print(f"⚠️  Error while waiting for models: {e}")
                # Continue waiting
        else:
            # Timeout reached
            print("⏰ Timeout waiting for models to load, attempting direct load...")
            try:
                initialize_models()
                with MODEL_LOADING_LOCK:
                    MODELS_LOADED = True
                    MODELS_LOADING = False
                    MODEL_LOAD_ERROR = None
                print("✅ TalkNet models loaded successfully\n")
            except Exception as e:
                print(f"❌ Failed to load TalkNet models: {e}")
                print("⚠️  Video processing may fail without TalkNet\n")
    else:
        # Models not loading and not loaded - start loading now
        print("📦 TalkNet models not loaded, loading now...")
        try:
            initialize_models()
            with MODEL_LOADING_LOCK:
                MODELS_LOADED = True
                MODELS_LOADING = False
                MODEL_LOAD_ERROR = None
            print("✅ TalkNet models loaded successfully\n")
        except Exception as e:
            print(f"❌ Failed to load TalkNet models: {e}")
            print("⚠️  Video processing may fail without TalkNet\n")
    
    # Ensure we have model references even if there were errors above
    try:
        from reelsfy_folder.reelsfy import GLOBAL_TALKNET, GLOBAL_TALKNET_DET
    except ImportError:
        print("⚠️  Warning: Could not import TalkNet models, continuing anyway...")
        GLOBAL_TALKNET = None
        GLOBAL_TALKNET_DET = None
    
    # 1) Find the video's UUID in the videos table
    print("\n" + "="*60)
    print("STAGE: Find video UUID in database")
    print("="*60)
    print("▶️  RUNNING: Find video UUID in database (auto-continue mode)\n")
    

    resp = supabase.table("videos").select("id").eq("file_url", file_key).execute()
    if not resp.data or len(resp.data) == 0:
        print(f"[ERROR] No video row found for file_key: {file_key}")
        return
    video_id = resp.data[0]["id"]
    
    # Initialize progress tracking
    update_progress(video_id, 0, "מתחיל עיבוד...")

    # 1) Download original video from Supabase Storage
    print("\n" + "="*60)
    print("STAGE: Download original video from Supabase Storage")
    print("="*60)
    print("▶️  RUNNING: Download original video from Supabase Storage (auto-continue mode)\n")
    
    update_progress(video_id, 5, "מוריד סרטון מקורי...")
    local_in = os.path.basename(file_key)
    data = supabase.storage.from_(bucket).download(file_key)
    if isinstance(data, bytes):
        with open(local_in, "wb") as f:
            f.write(data)
    else:
        content = data.read() if hasattr(data, "read") else data
        with open(local_in, "wb") as f:
            f.write(content)
    print(f"Downloaded: {local_in}")

    # 2) Process with reelsfy logic (stops before burning subtitles)
    print("\n" + "="*60)
    print("STAGE: Process video with reelsfy (transcription, segmentation, cropping, silence removal)")
    print("="*60)
    print("▶️  RUNNING: Process video with reelsfy (auto-continue mode)\n")
    
    update_progress(video_id, 10, "מתחיל עיבוד ראשוני...")
    
    # Use explicit is_short_video flag from frontend (already passed to this function)
    print(f"Short video mode: {is_short_video}")
    
    try:
        process_video_file(local_in, out_dir="tmp", settings=settings, video_id=video_id, 
                          progress_callback=update_progress, is_short_video=is_short_video)
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"❌ ERROR: Video processing failed!")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        # Update video status to failed
        try:
            supabase.table("videos").update({
                "status": "failed",
                "progress": 0
            }).eq("id", video_id).execute()
            print(f"✅ Updated video status to 'failed' in database")
        except Exception as db_error:
            print(f"⚠️  Could not update video status: {db_error}")
        
        # Don't continue with uploads/cleanup if processing failed
        print(f"❌ Aborting processing pipeline due to error")
        return

    # Create consistent folder name in storage (e.g. user123/filename-without-ext)
    # The file_key format is: user_id/timestamp.mp4 (e.g., 6ff414f2-0f7d-463a-af6c-f86dc1073d2e/1760477721689.mp4)
    file_key_no_ext = os.path.splitext(file_key)[0]  # remove .mp4 extension
    
    # Extract user ID and timestamp separately
    if '/' in file_key_no_ext:
        user_id, video_folder_name = file_key_no_ext.split('/', 1)
    else:
        user_id = bucket
        video_folder_name = file_key_no_ext
    
    # Full path for uploading (includes user_id)
    video_output_dir = file_key_no_ext  # e.g., user_id/timestamp
    
    vid_name_no_mp4 = local_in.split(".")[0]
    content_txt_path = f"results/{vid_name_no_mp4}/content.txt"

    # 3) Upload processed results (videos + SRTs, but NOT final_xxx.mp4)
    print("\n" + "="*60)
    print("STAGE: Upload processed results to Supabase Storage")
    print("="*60)
    print("▶️  RUNNING: Upload processed results to Supabase Storage (auto-continue mode)\n")
    
    update_progress(video_id, 85, "מעלה קבצים מעובדים...")
    print("Uploading files to processed-videos bucket...")
    
    if os.path.exists(content_txt_path):
        supabase.storage.from_("processed-videos").upload(
            f"{video_output_dir}/content.txt", content_txt_path
        )
        print("Uploaded content.txt")
    else:
        print(f"Did not find content.txt in {content_txt_path}")

    for fname in os.listdir("tmp"):
        # For short videos: only upload output_cropped files (skip output_croppedwithoutcutting)
        # For regular videos: upload all output_cropped files
        # Note: SRT files are now word-level only (chunking happens in frontend)
        if is_short_video:
            # Short video: only upload the essential files
            if (fname == "output_cropped000.mp4" or 
                fname == "output_cropped000.srt"):
                local_path = os.path.join("tmp", fname)
                dest_path = f"{video_output_dir}/{fname}"
                supabase.storage.from_("processed-videos").upload(dest_path, local_path)
                print(f"Uploaded {fname}")
        else:
            # Regular video: upload all output_cropped files
            # SRT files are word-level (will be chunked in frontend)
            if (fname == "content.txt" or 
                fname.startswith("output_cropped") and fname.endswith(".mp4") or
                fname.startswith("output_cropped") and fname.endswith(".srt") or
                fname.startswith("output_nosilence") and fname.endswith(".srt")):
                local_path = os.path.join("tmp", fname)
                dest_path = f"{video_output_dir}/{fname}"
                supabase.storage.from_("processed-videos").upload(dest_path, local_path)
                print(f"Uploaded {fname}")
    print("Finished uploading files")

    # 4) Read content.txt and create rows for shorts (aligned by index)
    print("\n" + "="*60)
    print("STAGE: Prepare database records for shorts")
    print("="*60)
    print("▶️  RUNNING: Prepare database records for shorts (auto-continue mode)\n")
    
    update_progress(video_id, 90, "מכין רשומות מסד נתונים...")
    shorts_to_insert = []
    
    if is_short_video:
        # Short video: create a single short record directly
        print("Short video mode - creating single short record")
        
        final_name = "final_000.mp4"
        srt_name = "output_cropped000.srt"
        
        # Read SRT content
        srt_content = None
        srt_local = os.path.join("tmp", srt_name)
        if os.path.exists(srt_local):
            try:
                with open(srt_local, "r", encoding="utf-8") as sf:
                    srt_content = sf.read()
            except Exception as e:
                print("Failed reading SRT:", e)
        
        # Read title and description from content.txt (generated by GPT)
        title = "סרטון קצר"  # Default title
        description = ""
        if os.path.exists(content_txt_path):
            try:
                with open(content_txt_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    title = content.get("title", "סרטון קצר")
                    description = content.get("description", "")
                    print(f"Read title from GPT: {title}")
                    print(f"Read description from GPT: {description}")
            except Exception as e:
                print(f"Failed reading title/description from content.txt: {e}")
        
        shorts_to_insert.append({
            "video_id": video_id,
            "filename": final_name,
            "title": title,
            "description": description,
            "start_time": "0",
            "end_time": str(settings.get('maxClipLength', 0)),  # Full video duration
            "duration": str(settings.get('maxClipLength', 0)),
            "file_url": f"{video_output_dir}/{final_name}",
            "srt_content": srt_content,
            "original_video_filename": local_in,
            "user_folder_id": user_id,
            "video_folder_name": video_folder_name,
        })
        print(f"Prepared DB row for short video: {final_name} with title: {title}")
        
    elif os.path.exists(content_txt_path):
        # Regular video: read segments from content.txt
        with open(content_txt_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        segments = content.get("segments", [])

        for i, seg in enumerate(segments):
            start_time  = str(seg.get("start_time", ""))
            end_time    = str(seg.get("end_time", ""))
            duration    = str(seg.get("duration", ""))
            title       = seg.get("title", None)         # Hebrew
            description = seg.get("description", "")     # Hebrew

            # we know our output names from reelsfy.py
            final_name   = f"final_{i:03}.mp4"
            srt_name     = f"output_cropped{i:03}.srt"  # Updated to match actual filename

            # best-effort read of retimed SRT content (includes <color> tags from GPT)
            srt_content = None
            srt_local = os.path.join("tmp", srt_name)
            if os.path.exists(srt_local):
                try:
                    with open(srt_local, "r", encoding="utf-8") as sf:
                        srt_content = sf.read()
                except Exception as e:
                    print("Failed reading SRT:", e)

            shorts_to_insert.append({
                "video_id": video_id,
                "filename": final_name,
                "title": title,
                "description": description,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "file_url": f"{video_output_dir}/{final_name}",
                "srt_content": srt_content,  # keep a copy in DB
                "original_video_filename": local_in,
                "user_folder_id": user_id,  # User ID (e.g., 6ff414f2-0f7d-463a-af6c-f86dc1073d2e)
                "video_folder_name": video_folder_name,  # Only the timestamp part (e.g., 1760477721689)
            })
            print(f"Prepared DB row for {final_name}")
    
    print(f"Prepared {len(shorts_to_insert)} shorts records")

    # 5) Insert rows into shorts table
    print("\n" + "="*60)
    print("STAGE: Insert shorts records into database")
    print("="*60)
    print("▶️  RUNNING: Insert shorts records into database (auto-continue mode)\n")
    
    if shorts_to_insert:
        supabase.table("shorts").insert(shorts_to_insert).execute()
        print(f"Inserted {len(shorts_to_insert)} shorts into database")
    else:
        print("No shorts to insert")

    # 6) Update video row to mark as processed
    print("\n" + "="*60)
    print("STAGE: Update video status to completed")
    print("="*60)
    print("▶️  RUNNING: Update video status to completed (auto-continue mode)\n")
    
    update_progress(video_id, 100, "הושלם!")
    supabase.table("videos").update({
        "status": "completed",
        "progress": 100
    }).eq("file_url", file_key).execute()
    print(f"Updated video status to completed")
    
    # 7) Cleanup temporary files
    print("\n" + "="*60)
    print("STAGE: Cleanup temporary files")
    print("="*60)
    print("▶️  RUNNING: Cleanup temporary files (auto-continue mode)\n")
    print("🧹 CLEANING UP TEMPORARY FILES")
    print("="*60)
    
    # Delete tmp/ folder
    if os.path.exists("tmp"):
        try:
            shutil.rmtree("tmp")
            print("�?Deleted tmp/ folder")
        except Exception as e:
            print(f"⚠️  Could not delete tmp/ folder: {e}")
    
    # Delete results/ folder
    if os.path.exists("results"):
        try:
            shutil.rmtree("results")
            print("�?Deleted results/ folder")
        except Exception as e:
            print(f"⚠️  Could not delete results/ folder: {e}")
    
    # Delete downloaded video file from main directory
    if os.path.exists(local_in):
        try:
            os.remove(local_in)
            print(f"�?Deleted downloaded video: {local_in}")
        except Exception as e:
            print(f"⚠️  Could not delete {local_in}: {e}")
    
    # Delete TalkNet temporary files in save/ folder and subdirectories
    # Structure: save/pyavi/ and save/pycrop/
    if os.path.exists("save"):
        try:
            deleted_count = 0
            
            # Delete files in save/ root
            for filename in os.listdir("save"):
                file_path = os.path.join("save", filename)
                if os.path.isfile(file_path) and filename.endswith(('.avi', '.wav', '.pckl')):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠️  Could not delete {file_path}: {e}")
            
            # Delete all files in save/pyavi/ subdirectory
            pyavi_dir = os.path.join("save", "pyavi")
            if os.path.exists(pyavi_dir):
                for filename in os.listdir(pyavi_dir):
                    file_path = os.path.join(pyavi_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            print(f"⚠️  Could not delete {file_path}: {e}")
            
            # Delete all files in save/pycrop/ subdirectory
            pycrop_dir = os.path.join("save", "pycrop")
            if os.path.exists(pycrop_dir):
                for filename in os.listdir(pycrop_dir):
                    file_path = os.path.join(pycrop_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            print(f"⚠️  Could not delete {file_path}: {e}")
            
            print(f"�?Deleted {deleted_count} TalkNet temporary files from save/ folder (including pyavi/ and pycrop/)")
        except Exception as e:
            print(f"⚠️  Could not access save/ folder: {e}")
    
    print("="*60)
    print("�?Cleanup complete!")
    print("="*60 + "\n")
    
    # Calculate and print total processing time
    processing_end_time = time.time()
    total_time = processing_end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*60)
    print("⏱️  TOTAL PROCESSING TIME")
    print("="*60)
    print(f"Total time: {minutes}m {seconds}s ({total_time:.2f} seconds)")
    print("="*60 + "\n")
    
    print(f"Finished run_reelsfy process.")


def clear_tmp_export_artifacts():
    """Remove stale export artifacts so per-clip exports don't mix with previous runs."""
    patterns = [
        "tmp/output_cropped*.mp4",
        "tmp/output_cropped*.srt",
        "tmp/output_croppedwithoutcutting*.mp4",
        "tmp/output_nosilence*.mp4",
        "tmp/output_nosilence*.srt",
        "tmp/final_*.mp4",
        "tmp/styling_data_*.json",
        "tmp/logo_*.png",
    ]

    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"⚠️  Could not remove {path}: {exc}")


def delete_supabase_file(storage_path: str):
    """Best-effort delete for a processed-videos file."""
    try:
        supabase.storage.from_("processed-videos").remove([storage_path])
        print(f"🗑️  Removed existing file (if any): {storage_path}")
    except Exception as exc:
        message = str(exc).lower()
        if "not found" in message or "does not exist" in message:
            print(f"ℹ️  No existing file to delete at {storage_path}")
        else:
            print(f"⚠️  Could not delete {storage_path}: {exc}")


def upload_with_overwrite(dest_path: str, payload, file_options=None, description: str = ""):
    """Upload to Supabase storage after ensuring previous versions are deleted."""
    delete_supabase_file(dest_path)

    options = dict(file_options or {})
    options.setdefault("upsert", "true")

    def _attempt_upload():
        return supabase.storage.from_("processed-videos").upload(dest_path, payload, options)

    try:
        _attempt_upload()
        print(f"📤 Uploaded {description or dest_path}")
    except Exception as exc:
        message = str(exc)
        if "Duplicate" in message or "409" in message or "already exists" in message.lower():
            print(f"⚠️  Duplicate detected for {dest_path}, retrying after forced delete...")
            delete_supabase_file(dest_path)
            _attempt_upload()
            print(f"📤 Uploaded {description or dest_path} after retry")
        else:
            raise


def run_export_processing(
    user_folder_id: str, 
    video_folder_name: str, 
    user_email: str, 
    video_id: str,
    export_data: dict
):
    """
    Handle export mode - receive styling data via HTTP and create final videos.
    
    export_data: {
        'videoId': str,
        'userFolderId': str,
        'videoFolderName': str,
        'shorts': [
            {
                'shortId': str,
                'shortIndex': int,
                'filename': str,
                'editedSrtContent': str,  # SRT with color tags
                'globalTextFormatting': {...},
                'individualBoxFormatting': [...],
                'textboxPosition': {...},
                'logo': {...},
                'music': {...}
            }
        ]
    }
    """
    print(f"Starting export processing for video {video_id}")
    print(f"Export data received for {len(export_data.get('shorts', []))} shorts")
    
    # Debug: Print the first short's data to see what fields are available
    if export_data.get('shorts'):
        first_short = export_data.get('shorts')[0]
        print(f"📋 First short data keys: {first_short.keys()}")
        print(f"📋 First short data: {first_short}")
    
    try:
        # Ensure working directories exist
        os.makedirs("tmp", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        clear_tmp_export_artifacts()
        
        # Use folder info from parameters (sent from frontend)
        video_output_dir = f"{user_folder_id}/{video_folder_name}"
        
        print(f"📁 Processing folders:")
        print(f"   User Folder: {user_folder_id}")
        print(f"   Video Folder: {video_folder_name}")
        print(f"   Output Path: {video_output_dir}")
        
        # 3) Download the pre-processed video files and save styling data locally
        print("Downloading pre-processed video files and saving styling data...")
            
        for short_data in export_data.get('shorts', []):
            short_index = short_data.get('shortIndex', 0)
            
            # Download the silence-cut video (output_cropped{xxx}.mp4)
            silence_video_key = f"{video_output_dir}/output_cropped{short_index:03}.mp4"
            print(f"Debug: Attempting to download from '{silence_video_key}'")
            try:
                video_data = supabase.storage.from_("processed-videos").download(silence_video_key)
                local_video_path = f"tmp/output_cropped{short_index:03}.mp4"
                
                if isinstance(video_data, bytes):
                    with open(local_video_path, "wb") as f:
                        f.write(video_data)
                else:
                    content = video_data.read() if hasattr(video_data, "read") else video_data
                    with open(local_video_path, "wb") as f:
                        f.write(content)
                
                print(f"Downloaded {local_video_path}")
                
                # Save the edited SRT content (from frontend, includes <color> tags)
                edited_srt_content = short_data.get('editedSrtContent', '')
                local_srt_path = f"tmp/output_nosilence{short_index:03}.srt"
                
                if edited_srt_content:
                    with open(local_srt_path, "w", encoding='utf-8') as f:
                        f.write(edited_srt_content)
                    print(f"Saved edited SRT to {local_srt_path}")
                else:
                    print(f"Warning: No SRT content provided for short {short_index}")
                
                # 4) Save styling data to local file (from HTTP request, not storage)
                styling_data = {
                    "globalTextFormatting": short_data.get("globalTextFormatting", {}),
                    "individualBoxFormatting": short_data.get("individualBoxFormatting", []),
                    "textboxPosition": short_data.get("textboxPosition", {}),
                    "logo": short_data.get("logo", {}),
                    "music": short_data.get("music", {})
                }
                
                styling_file = f"tmp/styling_data_{short_index:03}.json"
                with open(styling_file, 'w', encoding='utf-8') as f:
                    json.dump(styling_data, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Saved styling data to {styling_file}")
                print(f"   - Global formatting: fontSize={styling_data['globalTextFormatting'].get('fontSize')}, color={styling_data['globalTextFormatting'].get('color')}")
                print(f"   - Logo: exists={styling_data['logo'].get('exists')}, position={styling_data['logo'].get('position')}, opacity={styling_data['logo'].get('opacity')}")
                print(f"   - Music: track={styling_data['music'].get('track')}, volume={styling_data['music'].get('volume')}")
                print(f"   - Textbox: xPct={styling_data['textboxPosition'].get('xPct')}, yPct={styling_data['textboxPosition'].get('yPct')}")
                
                # Log detailed textbox position data
                textbox_pos = styling_data['textboxPosition']
                if textbox_pos:
                    print(f"🎯 [API] Received Textbox Position Data:")
                    print(f"   xPct: {textbox_pos.get('xPct')}%")
                    print(f"   yPct: {textbox_pos.get('yPct')}%")
                    print(f"   wPct: {textbox_pos.get('wPct')}%")
                    print(f"   hPct: {textbox_pos.get('hPct')}%")
                else:
                    print(f"⚠️ [API] No textbox position data received for short {short_index}")
                
                # 5) Download logo if exists
                logo_url = styling_data.get("logo", {}).get("url", "")
                if logo_url:
                    try:
                        logo_data = supabase.storage.from_("processed-videos").download(logo_url)
                        local_logo_path = f"tmp/logo_{short_index:03}.png"
                        
                        if isinstance(logo_data, bytes):
                            with open(local_logo_path, "wb") as f:
                                f.write(logo_data)
                        else:
                            content = logo_data.read() if hasattr(logo_data, "read") else logo_data
                            with open(local_logo_path, "wb") as f:
                                f.write(content)
                        
                        # Update styling data with local logo path
                        styling_data["logo"]["url"] = local_logo_path
                        with open(styling_file, 'w', encoding='utf-8') as f:
                            json.dump(styling_data, f, indent=2, ensure_ascii=False)
                        
                        print(f"Downloaded logo to {local_logo_path}")
                    except Exception as e:
                        print(f"Warning: Could not download logo for short {short_index}: {e}")
                
            except Exception as e:
                print(f"Error downloading video for short {short_index}: {e}")
                continue
        
        # 6) Process with export mode (this will create final_xxx.mp4 files)
        print("Processing videos with export mode (burn subtitles, add logos, add music)...")
        # We don't need to pass a file path since we're working with already downloaded files
        try:
            process_export_file("", out_dir="tmp")
            print("Finished processing export files")
        except Exception as e:
            import traceback
            print(f"\n{'='*60}")
            print(f"❌ ERROR: Export processing failed!")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            
            # Update video status to failed
            try:
                supabase.table("videos").update({
                    "status": "failed",
                    "export_ready": False
                }).eq("id", video_id).execute()
                print(f"✅ Updated video status to 'failed' in database")
            except Exception as db_error:
                print(f"⚠️  Could not update video status: {db_error}")
            
            return
        
        # 7) Upload final videos and edited SRT files
        print("Uploading final videos to Supabase Storage...")
        
        # Build a mapping of shortIndex to short data for easy lookup
        shorts_map = {short_data.get('shortIndex', 0): short_data for short_data in export_data.get('shorts', [])}
        
        for fname in os.listdir("tmp"):
            if fname.startswith("final_") and fname.endswith(".mp4"):
                local_path = os.path.join("tmp", fname)
                
                # Extract short index from filename (final_XXX.mp4)
                try:
                    short_index = int(fname.replace("final_", "").replace(".mp4", ""))
                    short_data = shorts_map.get(short_index, {})
                    
                    print(f"📝 Processing short index {short_index}:")
                    print(f"   Short data: {short_data}")
                    
                    # Get the short title or use default
                    short_title = short_data.get('title', f'קליפ {short_index + 1}')
                    print(f"   Title from data: {short_data.get('title', 'NOT FOUND')}")
                    print(f"   Using title: {short_title}")
                    
                    # Clean the title to be filesystem-safe
                    safe_title = re.sub(r'[<>:"/\\|?*]', '', short_title)
                    safe_title = safe_title.strip() or f'קליפ {short_index + 1}'
                    
                    # Create filename with title
                    video_filename = f"{safe_title}.mp4"
                    dest_path = f"{video_output_dir}/{video_filename}"
                    
                    upload_with_overwrite(
                        dest_path,
                        local_path,
                        description=f"final video {video_filename} (from {fname})"
                    )
                    
                except ValueError as e:
                    print(f"⚠️  Could not parse short index from filename {fname}: {e}")
                    # Fall back to original filename
                    dest_path = f"{video_output_dir}/{fname}"
                    supabase.storage.from_("processed-videos").upload(dest_path, local_path)
                    print(f"Uploaded final video: {fname}")
        
        # Upload edited SRT files with title-based names
        for short_data in export_data.get('shorts', []):
            short_index = short_data.get('shortIndex', 0)
            edited_srt_content = short_data.get('editedSrtContent')
            if edited_srt_content:
                # Get the short title or use default
                short_title = short_data.get('title', f'קליפ {short_index + 1}')
                
                # Clean the title to be filesystem-safe
                safe_title = re.sub(r'[<>:"/\\|?*]', '', short_title)
                safe_title = safe_title.strip() or f'קליפ {short_index + 1}'
                
                # Create SRT filename with title
                srt_filename = f"{safe_title}.srt"
                srt_key = f"{video_output_dir}/{srt_filename}"
                
                try:
                    upload_with_overwrite(
                        srt_key,
                        edited_srt_content.encode('utf-8'),
                        {"content-type": "text/plain"},
                        description=f"SRT {srt_filename}"
                    )
                except Exception as e:
                    print(f"Warning: Could not upload edited SRT {srt_filename}: {e}")
        print("Finished uploading final videos")
        
        # 8) Update video status to completed and set export_ready flag
        print("Updating video status to completed...")
        supabase.table("videos").update({
            "status": "completed",
            "export_ready": True  # NEW: Notify frontend that export is ready
        }).eq("id", video_id).execute()
        print(f"�?Export processing completed for video {video_id}")
        print(f"   Frontend will be notified via real-time subscription")
        
    except Exception as e:
        print(f"Error in export processing: {e}")
        # Update video status to completed (or use a valid status from your enum)
        supabase.table("videos").update({"status": "completed"}).eq("id", video_id).execute()
