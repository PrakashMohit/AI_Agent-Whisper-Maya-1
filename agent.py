import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import subprocess
import torch

from transformers import pipeline


# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- LOAD MODELS ONCE ----------------

print("Loading Whisper...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=0 if DEVICE == "cuda" else -1,
    generate_kwargs={
        "language": "en",
        "task": "transcribe",
        "temperature": 0.0,
    }
)


# ---------------- PIPER TTS ----------------

PIPER_EXE = r"D:\AI assistant\piper\piper.exe"
MODEL = r"D:\AI assistant\piper\models\en_US-amy-medium.onnx"

def piper_tts(text, out_path="reply.wav"):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as f:
        f.write(text)
        txt_path = f.name

    subprocess.run(
        [
            PIPER_EXE,
            "--model", MODEL,
            "--output_file", out_path
        ],
        stdin=open(txt_path, "r"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    os.remove(txt_path)


# ---------------- FUNCTIONS ----------------

def record_audio():
    print("Speak now...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio = audio.squeeze()
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(tmp.name, SAMPLE_RATE, (audio * 32767).astype("int16"))
    return tmp.name


def ollama_reason(user_text):
    prompt = (
        "You are a calm, concise personal assistant.\n"
        "Your name is Friday.\n"
        "Your are a Female.\n"
        "If asked your name, reply: 'My name is Friday.'\n"
        "Reply briefly and naturally.\n\n"
        f"User: {user_text}\nAssistant:"
    )

    result = subprocess.check_output(
        ["ollama", "run", "gemma3:4b", prompt],
        text=True
    )
    return result.strip()


def play_audio(path):
    subprocess.run(
        ["ffplay", "-autoexit", "-nodisp", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# ---------------- MAIN LOOP ----------------

print("\n🎙️ Voice agent ready.")

while True:
    input("\nPress ENTER to talk (Ctrl+C to quit)")

    audio_path = record_audio()

    print("Transcribing...")
    result = asr(audio_path)
    os.remove(audio_path)

    user_text = result["text"].strip()
    print("You:", user_text)

    if not user_text:
        print("⚠️ No speech detected.")
        continue

    reply = ollama_reason(user_text)
    print("Agent:", reply)

    piper_tts(reply)
    play_audio("reply.wav")
    os.remove("reply.wav")

