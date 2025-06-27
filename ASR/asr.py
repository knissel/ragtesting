import streamlit as st
import nemo.collections.asr as nemo_asr
import torch
import torchaudio
import os
import io
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
from pydub.utils import make_chunks

# --- Constants & Temp Dir ---
PAGE_TITLE = "🔊 Parakeet ASR Transcription"
TEMP_DIR = "temp_audio"
CHUNK_MS = 8 * 60 * 1000  # 8 minutes in ms
os.makedirs(TEMP_DIR, exist_ok=True)


# --- Session State Init ---
for key in ("mic_transcription", "upload_transcription", "last_audio_bytes"):
    if key not in st.session_state:
        st.session_state[key] = None


# --- Page Config ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.write("Upload an audio file or record live — transcribed by NVIDIA Parakeet.")


# --- Load ASR Model ---
@st.cache_resource
def load_asr_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner(f"Loading ASR model on {device.upper()}…"):
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        ).to(device)

    # Warm-up pass
    dummy = torch.randn((1, 16000 * 5), device=device)  # 5 seconds of noise
    torchaudio.save(os.path.join(TEMP_DIR, "warmup.wav"), dummy.cpu(), 16000)
    with st.spinner("Warming up model…"):
        model.transcribe([os.path.join(TEMP_DIR, "warmup.wav")])
    os.remove(os.path.join(TEMP_DIR, "warmup.wav"))

    return model

asr_model = load_asr_model()
if not asr_model:
    st.error("Failed to load ASR model.")
    st.stop()


# --- Transcription Logic ---
def transcribe_audio(file_path: str) -> str:
    audio = AudioSegment.from_file(file_path)
    total_ms = len(audio)
    texts = []

    if total_ms > CHUNK_MS:
        chunks = make_chunks(audio, CHUNK_MS)
        progress = st.progress(0)
        for idx, chunk in enumerate(chunks, start=1):
            chunk_path = os.path.join(TEMP_DIR, f"chunk_{idx}.wav")
            chunk.export(chunk_path, format="wav")
            hyp = asr_model.transcribe([chunk_path])[0]
            texts.append(hyp.text)
            os.remove(chunk_path)
            progress.progress(idx / len(chunks))
    else:
        hyp = asr_model.transcribe([file_path])[0]
        texts.append(hyp.text)

    return " ".join(texts)


# --- UI Tabs ---
tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Record Mic"])

with tab1:
    st.header("Upload an Audio File")
    uploaded = st.file_uploader("Choose .wav, .mp3 or .flac", type=["wav", "mp3", "flac"])
    if uploaded:
        st.audio(uploaded)
        if st.button("Transcribe Upload"):
            temp_path = os.path.join(TEMP_DIR, uploaded.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("Transcribing upload…"):
                text = transcribe_audio(temp_path)
            os.remove(temp_path)

            st.session_state.upload_transcription = text

    if st.session_state.upload_transcription:
        st.subheader("Transcription Result")
        st.text_area("Result", st.session_state.upload_transcription, height=200)


with tab2:
    st.header("Record from Microphone")
    st.write("Click the mic icon to start/stop recording.")
    audio_bytes = audio_recorder(
        text="🎙 Record", recording_color="#e84242",
        neutral_color="#6aa36f", icon_size="3x"
    )

    if audio_bytes:
        st.audio(audio_bytes)

        # reset transcription on new recording
        if st.session_state.last_audio_bytes != audio_bytes:
            st.session_state.mic_transcription = None
            st.session_state.last_audio_bytes = audio_bytes

        if st.button("Transcribe Recording"):
            mic_path = os.path.join(TEMP_DIR, "mic_recording.wav")
            with open(mic_path, "wb") as f:
                f.write(audio_bytes)

            # ←—— DOWN-MIX TO MONO HERE ——
            audio = AudioSegment.from_file(mic_path)
            if audio.channels > 1:
                audio = audio.set_channels(1)
                audio.export(mic_path, format="wav")

            with st.spinner("Transcribing recording…"):
                text = transcribe_audio(mic_path)

            os.remove(mic_path)
            st.session_state.mic_transcription = text

    if st.session_state.mic_transcription:
        st.subheader("Transcription Result")
        st.text_area("Result", st.session_state.mic_transcription, height=200)
