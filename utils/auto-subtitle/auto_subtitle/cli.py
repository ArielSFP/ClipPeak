import os
import ffmpeg
import whisper
import torch  # <-- Import torch to check for CUDA
import argparse
import warnings
import tempfile
import sys
from typing import Iterator, TextIO

# --- Utility Functions (Unchanged) ---
def filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def write_srt(result: Iterator[dict], file: TextIO):
    for i, segment in enumerate(result, start=1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip().replace('-->', '->')
        file.write(f"{i}\n")
        file.write(f"{start_time} --> {end_time}\n")
        file.write(f"{text}\n\n")

def format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000; milliseconds %= 3_600_000
    minutes = milliseconds // 60_000; milliseconds %= 60_000
    seconds = milliseconds // 1_000; milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def escape_path_for_ffmpeg_filter(path):
    # This function remains crucial for Windows paths in filters
    return os.path.abspath(path).replace('\\', '/').replace(':', '\\:')


def main():
    # --- Argparse setup is unchanged ---
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str, help="paths to video files to transcribe")
    parser.add_argument("--model", default="turbo", choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=False, help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False, help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--force", type=str2bool, default=False, help="Force re-processing even if output files already exist.")
    parser.add_argument("--verbose", type=str2bool, default=False, help="whether to print out the progress and debug messages")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], help="What is the origin language of the video? If unset, it is detected automatically.")
    args = parser.parse_args().__dict__
    
    model_name: str = args.pop("model"); output_dir: str = args.pop("output_dir"); output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only"); language: str = args.pop("language"); force: bool = args.pop("force")
    
    os.makedirs(output_dir, exist_ok=True)

    # --- WHISPER ON GPU ---
    # Check if CUDA is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device '{device}' for Whisper transcription.")
    model = whisper.load_model(model_name, device=device)

    # --- The rest of the setup is the same ---
    if model_name.endswith(".en"):
        warnings.warn(f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    elif language != "auto":
        args["language"] = language
        
    audios = get_audio(args.pop("video"))
    subtitles_info = get_subtitles(audios, output_srt or srt_only, output_dir, lambda audio_path: model.transcribe(audio_path, **args), force=force)
    
    if srt_only:
        print("SRT-only mode enabled. Skipping video creation."); return

    for path, srt_path in subtitles_info.items():
        out_path = os.path.join(output_dir, f"{filename(path)}_subtitled.mp4")
        if not force and os.path.exists(out_path):
            print(f"Output video already exists: {os.path.abspath(out_path)}. Skipping."); continue

        print(f"Adding subtitles to {filename(path)} using the 'concat' method...")

        # --- FFmpeg with 'concat' structure and GPU encoding ---
        
        # 1. Prepare paths and styles (this part is still crucial)
        escaped_srt_path = escape_path_for_ffmpeg_filter(srt_path)
        style_str = "Alignment=2,MarginV=40,MarginL=55,MarginR=55,Fontname=Arial,Fontsize=11,PrimaryColour=&H00d7ff,Outline=1,Shadow=1,BorderStyle=1"
        # The comma escaping is needed for the force_style option
        escaped_style_str = style_str.replace(',', '\\,')

        # 2. Define input streams (decoding on CPU is safer for filters)
        input_stream = ffmpeg.input(path, hwaccel='cuda')
        video = input_stream.video
        audio = input_stream.audio

        # 3. Apply the subtitles filter to the video stream
        subtitled_video = video.filter(
            'subtitles',
            escaped_srt_path,
            force_style=escaped_style_str
        )
        
        # 4. Build the graph using concat, then apply output options
        # This structure explicitly combines the filtered video with the original audio.
        (
            ffmpeg
            .concat(subtitled_video, audio, v=1, a=1)
            .output(
                out_path, 
                vcodec='h264_nvenc',  # Use GPU for encoding
                acodec='copy',         # Copy audio without re-encoding
                preset='llhp'
            )
            .run(quiet=False, overwrite_output=True)
        )
        
        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


# --- The get_audio and get_subtitles functions remain the same ---
def get_audio(paths):
    temp_dir = tempfile.gettempdir(); audio_paths = {}
    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")
        try:
            ffmpeg.input(path).output(output_path, acodec="pcm_s16le", ac=1, ar="16k").run(quiet=True, overwrite_output=True)
            audio_paths[path] = output_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio from {path}:", file=sys.stderr); print(e.stderr.decode(), file=sys.stderr)
    return audio_paths

def get_subtitles(audio_paths: dict, output_srt: bool, output_dir: str, transcribe: callable, force: bool):
    subtitles_path = {}
    for path, audio_path in audio_paths.items():
        srt_filename = f"{filename(path)}.srt"
        srt_dir = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_dir, srt_filename)
        if not force and os.path.exists(srt_path):
            print(f"Found existing SRT file: {srt_path}. Skipping transcription.")
            subtitles_path[path] = srt_path
            os.remove(audio_path); continue
        print(f"Generating subtitles for {filename(path)}... This might take a while.")
        warnings.filterwarnings("ignore"); result = transcribe(audio_path); warnings.filterwarnings("default")
        with open(srt_path, "w", encoding="utf-8") as srt: write_srt(result["segments"], file=srt)
        subtitles_path[path] = srt_path
        os.remove(audio_path)
    return subtitles_path


if __name__ == '__main__':
    main()