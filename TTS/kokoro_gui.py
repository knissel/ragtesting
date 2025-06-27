import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import numpy as np # Required for audio data handling
from kokoro import KPipeline
import soundfile as sf
import torch
# Optional: for playing sound automatically
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False
    print("playsound library not found. Audio will not play automatically. Install with: pip install playsound")


# --- Kokoro TTS Configuration ---
LANG_CODE = 'a' # Default to American English

# Updated list of American English voices
american_english_voices = [
    "af_heart",   # 🚺❤️
    "af_alloy",   # 🚺
    "af_aoede",   # 🚺
    "af_bella",   # 🚺🔥
    "af_jessica", # 🚺
    "af_kore",    # 🚺
    "af_nicole",  # 🚺🎧
    "af_nova",    # 🚺
    "af_river",   # 🚺
    "af_sarah",   # 🚺
    "af_sky",     # 🚺
    "am_adam",    # 🚹
    "am_echo",    # 🚹
    "am_eric",    # 🚹
    "am_fenrir",  # 🚹
    "am_liam",    # 🚹
    "am_michael", # 🚹
    "am_onyx",    # 🚹
    "am_puck",    # 🚹
    "am_santa"    # 🚹
]

AVAILABLE_VOICES = {
    'a': american_english_voices,
    # 'b': ["bf_ava", "bm_noah"], # Example: British English (keep if you have these)
    # 'e': ["ef_valentina", "em_mateo"], # Example: Spanish (keep if you have these)
    # 'j': ["jf_madoka", "jm_takeru"], # Example: Japanese (requires misaki[ja]) (keep if you have these)
    # Add other languages and their voices here if needed
}
DEFAULT_VOICES_LIST = AVAILABLE_VOICES.get(LANG_CODE, ["af_heart"]) # Default to first in list or af_heart

pipeline = None
pipeline_init_error_message = None

def initialize_pipeline_thread():
    global pipeline, pipeline_init_error_message
    try:
        print("Initializing Kokoro KPipeline...")
        pipeline = KPipeline(lang_code=LANG_CODE)
        print("Kokoro KPipeline initialized successfully.")
        root.after(0, on_pipeline_initialized)
    except Exception as e:
        pipeline_init_error_message = (
            f"Failed to initialize Kokoro KPipeline: {e}\n\n"
            "Common troubleshooting steps:\n"
            "1. Ensure 'espeak-ng' is installed correctly.\n"
            "2. Add the 'espeak-ng' installation directory (e.g., 'C:\\Program Files\\eSpeak NG' on Windows) "
            "to your system's PATH environment variable.\n"
            "3. You might need to restart your computer or terminal/IDE after updating PATH.\n"
            "4. Ensure Microsoft C++ Redistributables are up to date (for Windows espeak-ng).\n"
            "5. If using non-English languages like Japanese or Chinese, ensure you've installed the "
            "necessary 'misaki' components (e.g., 'pip install misaki[ja]')."
        )
        print(pipeline_init_error_message)
        pipeline = None
        root.after(0, on_pipeline_init_failed)

def on_pipeline_initialized():
    status_label.config(text="Ready. Kokoro pipeline initialized.")
    generate_button.config(state=tk.NORMAL)
    if not DEFAULT_VOICES_LIST:
        messagebox.showwarning("Voice Configuration", f"No voices found for language code '{LANG_CODE}'. Please check AVAILABLE_VOICES in the script.")
        voice_dropdown.config(state=tk.DISABLED)

def on_pipeline_init_failed():
    status_label.config(text="Error: Kokoro pipeline failed to initialize.")
    generate_button.config(state=tk.DISABLED)
    if pipeline_init_error_message:
        messagebox.showerror("Pipeline Initialization Failed", pipeline_init_error_message)
    else:
        messagebox.showerror("Pipeline Initialization Failed", "An unknown error occurred during pipeline initialization. Check the console output for details.")


def generate_speech_task():
    """Handles the speech generation process in a separate thread with detailed logging."""
    global pipeline
    if pipeline is None:
        messagebox.showerror("Error", "Kokoro pipeline not initialized. Please check console for errors and restart the application.")
        root.after(0, lambda: generate_button.config(state=tk.NORMAL))
        root.after(0, lambda: status_label.config(text="Error: Pipeline not initialized."))
        return

    text_to_speak = text_area.get("1.0", tk.END).strip()
    selected_voice = voice_var.get()

    if not text_to_speak:
        messagebox.showwarning("Input Error", "Please enter some text to synthesize.")
        root.after(0, lambda: generate_button.config(state=tk.NORMAL))
        root.after(0, lambda: status_label.config(text="Ready."))
        return

    if not selected_voice or selected_voice == "No voices available":
        messagebox.showwarning("Input Error", "Please select a valid voice.")
        root.after(0, lambda: generate_button.config(state=tk.NORMAL))
        root.after(0, lambda: status_label.config(text="Ready."))
        return

    output_filename = ""
    audio_generated_successfully = False
    final_audio_data = None # Renamed to avoid confusion with audio_chunk

    try:
        print(f"Generating speech with voice: {selected_voice} for text: \"{text_to_speak[:100]}...\"")
        
        generator = pipeline(text_to_speak, voice=selected_voice)
        
        all_audio_segments = [] # Renamed from all_audio_chunks for clarity
        print("Starting to iterate through Kokoro pipeline generator...")
        segment_count = 0
        for i, (graphemes_segment, phonemes_segment, audio_chunk_data) in enumerate(generator): # Renamed audio_chunk to audio_chunk_data
            segment_count += 1
            print(f"--- Segment {i} ---")
            print(f"  Graphemes: '{graphemes_segment}'")
            print(f"  Phonemes: '{phonemes_segment}'")

            processed_audio_segment = None # Variable to hold the numpy array

            if audio_chunk_data is None:
                print(f"  Audio Data: None")
            elif isinstance(audio_chunk_data, torch.Tensor): # <--- CHECK FOR TORCH.TENSOR FIRST
                print(f"  Audio Data Type: torch.Tensor")
                print(f"  Audio Tensor Shape: {audio_chunk_data.shape}")
                print(f"  Audio Tensor dtype: {audio_chunk_data.dtype}")
                # Detach from graph, move to CPU (if on GPU), and convert to numpy
                processed_audio_segment = audio_chunk_data.detach().cpu().numpy()
                print(f"  Converted to NumPy. Shape: {processed_audio_segment.shape}, Size: {processed_audio_segment.size}")
            elif isinstance(audio_chunk_data, np.ndarray): # Check for numpy array
                print(f"  Audio Data Type: numpy.ndarray")
                processed_audio_segment = audio_chunk_data # Already a numpy array
                print(f"  Audio NumPy Shape: {processed_audio_segment.shape}, Size: {processed_audio_segment.size}")
            else:
                print(f"  Audio Data Type: {type(audio_chunk_data)} (Unexpected)")
                print(f"  Audio Data Value: {audio_chunk_data}")

            # Now work with processed_audio_segment (which should be a numpy array or None)
            if processed_audio_segment is not None and processed_audio_segment.size > 0:
                print(f"  Audio Segment Min/Max: {np.min(processed_audio_segment):.4f} / {np.max(processed_audio_segment):.4f}")
                all_audio_segments.append(processed_audio_segment)
            elif processed_audio_segment is not None: # It's a numpy array but size is 0
                 print(f"  Audio Segment is empty (size 0).")
            # else: audio_chunk_data was None or an unhandled type, already printed.

        print(f"Finished iterating through generator. Processed {segment_count} segments.")

        if not all_audio_segments:
            print("No valid audio segments were collected.")
            messagebox.showinfo("TTS Result", "No audio data generated. The input text might be empty, invalid for TTS, or too short. Check console for segment details.")
            audio_generated_successfully = False
        else:
            print(f"Collected {len(all_audio_segments)} audio segments. Concatenating...")
            final_audio_data = np.concatenate(all_audio_segments)
            print(f"Final audio data shape: {final_audio_data.shape}, size: {final_audio_data.size}")
            if final_audio_data.size == 0: # Check if concatenation resulted in empty array
                print("Concatenated audio data is empty.")
                messagebox.showinfo("TTS Result", "Generated audio data is empty after processing. Check console logs.")
                audio_generated_successfully = False
            else:
                audio_generated_successfully = True

        if audio_generated_successfully and final_audio_data is not None: # final_audio_data should not be None if True
            output_filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")],
                title="Save Speech As...",
                initialfile="kokoro_tts_output.wav"
            )

            if output_filename:
                sf.write(output_filename, final_audio_data, 24000)
                messagebox.showinfo("Success", f"Audio saved to {output_filename}")
                status_label.config(text=f"Saved to {output_filename}")

                if PLAYSOUND_AVAILABLE:
                    try:
                        threading.Thread(target=playsound, args=(output_filename,), daemon=True).start()
                    except Exception as e_play:
                        print(f"Error playing sound with playsound: {e_play}")
                        messagebox.showwarning("Playback Warning", f"Could not play audio automatically: {e_play}")
            else:
                status_label.config(text="Save cancelled. Audio generated but not saved.")
        elif not audio_generated_successfully: # This covers cases where it failed before save dialog
             status_label.config(text="Ready (no valid audio generated).")


    except Exception as e:
        import traceback
        error_message = f"An error occurred during speech generation: {e}\n\nTraceback:\n{traceback.format_exc()}"
        messagebox.showerror("TTS Error", f"An error occurred during speech generation: {e}. Check console for full traceback.")
        status_label.config(text="Error during generation.")
        print(error_message)
    finally:
        root.after(0, lambda: generate_button.config(state=tk.NORMAL))
        current_status = status_label.cget("text")
        # Only reset to "Ready" if not saved, errored, or specifically set to (no audio generated) etc.
        if "Saved to" not in current_status and \
           "Error" not in current_status and \
           "no valid audio generated" not in current_status and \
           "Save cancelled" not in current_status:
            root.after(0, lambda: status_label.config(text="Ready."))

def start_generate_speech_thread():
    generate_button.config(state=tk.DISABLED)
    status_label.config(text="Generating audio... Please wait.")
    tts_thread = threading.Thread(target=generate_speech_task, daemon=True)
    tts_thread.start()

# --- GUI Setup ---
root = tk.Tk()
root.title("Kokoro TTS GUI")
root.geometry("600x450")

input_frame = ttk.LabelFrame(root, text="Input Text")
input_frame.pack(padx=10, pady=10, fill="both", expand=True)

text_area = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=70, height=15, relief=tk.SOLID, borderwidth=1)
text_area.pack(padx=5, pady=5, fill="both", expand=True)
text_area.insert(tk.END, "Hello, this is a test of the Kokoro Text-to-Speech engine. [Kokoro](/kˈOkəɹO/) is quite fast!")

control_frame = ttk.LabelFrame(root, text="Controls")
control_frame.pack(padx=10, pady=(0, 5), fill="x")

ttk.Label(control_frame, text="Select Voice:").pack(side=tk.LEFT, padx=(5, 2), pady=5)
voice_var = tk.StringVar()
current_voices = DEFAULT_VOICES_LIST if DEFAULT_VOICES_LIST else ["No voices available"]
voice_var.set(current_voices[0])

voice_dropdown = ttk.OptionMenu(control_frame, voice_var, current_voices[0], *current_voices)
voice_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
if not DEFAULT_VOICES_LIST:
    voice_dropdown.config(state=tk.DISABLED)

generate_button = ttk.Button(control_frame, text="Generate & Save Speech", command=start_generate_speech_thread)
generate_button.pack(side=tk.LEFT, padx=5, pady=5)
generate_button.config(state=tk.DISABLED)

status_label = ttk.Label(root, text="Initializing Kokoro pipeline... Please wait.")
status_label.pack(pady=(0, 10), padx=10, fill="x", anchor="w")

print("Starting Kokoro Pipeline initialization in a background thread...")
init_thread = threading.Thread(target=initialize_pipeline_thread, daemon=True)
init_thread.start()

root.mainloop()
print("Application closed.")