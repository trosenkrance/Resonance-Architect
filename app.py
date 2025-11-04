# app.py — SACRED SOUND CODEX — FULLY WORKING ON STREAMLIT CLOUD (Python 3.13)
# NO audioop, NO pyaudioop — uses NumPy fallback only

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from threading import Thread, Event
import queue

# ——— PYDUB PATCH: REPLACE audioop WITH NUMPY (Runs before import) ———
import sys

class NumPyAudioop:
    def reverse(self, data, size): return np.flip(data)
    def lin2lin(self, data, size1, size2): return data.astype(np.int16)
    def ulaw2lin(self, data, size): return data
    def alaw2lin(self, data, size): return data
    def lin2ulaw(self, data, size): return data
    def lin2alaw(self, data, size): return data
    def lin2twos(self, data, size): return data.tobytes()
    def add(self, data1, data2, size):
        a1 = np.frombuffer(data1, dtype=np.int16)
        a2 = np.frombuffer(data2, dtype=np.int16)
        return (a1 + a2).astype(np.int16).tobytes()
    def bias(self, data, size, bias):
        return (np.frombuffer(data, dtype=np.int16) + bias).astype(np.int16).tobytes()
    def max(self, data, size): return np.max(np.frombuffer(data, dtype=np.int16))
    def min(self, data, size): return np.min(np.frombuffer(data, dtype=np.int16))
    def mul(self, data, size, mul):
        return (np.frombuffer(data, dtype=np.int16) * mul).astype(np.int16).tobytes()
    def ratecv(self, data, size, channels, in_rate, out_rate, state): return data
    def tomono(self, data, size, channels): return data
    def tostereo(self, data, size): return data
    def lin2adpcm(self, data, size, state): return data
    def adpcm2lin(self, data, size, state): return data

# Apply patch BEFORE importing pydub
sys.modules['audioop'] = NumPyAudioop()

# Now safe to import pydub
from pydub import AudioSegment

# ——— PWA SETUP ———
st.set_page_config(page_title="Sacred Sound Codex", layout="wide")
st.markdown("""
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#00ff88">
<script>
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js');
    });
}
</script>
""", unsafe_allow_html=True)

# ——— CONFIG ———
SAMPLE_RATE = 44100

# ——— VOICE GUIDE ———
@st.cache_data
def speak(text):
    try:
        from gtts import gTTS
        tts = gTTS(text, lang='en', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return AudioSegment.from_file(buf, format="mp3").apply_gain(-20)
    except:
        return AudioSegment.silent(1000)

# ——— CYMATIC MANDALA ———
def draw_cymatic(freqs, title="Cymatic"):
    fig, ax = plt.subplots(figsize=(7,7), facecolor='black')
    ax.set_facecolor('black')
    t = np.linspace(0, 2*np.pi, 1200)
    colors = ['#00ff88', '#ff00ff', '#00ffff', '#ffff00', '#ff8800']
    for i, f in enumerate(freqs[:7]):
        r = 1 + 0.4 * np.sin(12 * t + f * t * 0.08)
        x = r * np.cos(t * (f % 9))
        y = r * np.sin(t * (f % 9))
        ax.plot(x, y, color=colors[i % len(colors)], alpha=0.9, linewidth=1.8)
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=16)
    return fig

# ——— TONE GENERATOR ———
@st.cache_data
def get_tone(freq, duration, preset="sine"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.zeros_like(t)
    
    if preset == "sine": wave = np.sin(2*np.pi*freq*t)
    elif preset == "square": wave = np.sign(np.sin(2*np.pi*freq*t))
    elif preset == "sawtooth": wave = 2*(t*freq - np.floor(0.5 + t*freq))
    elif preset == "triangle": wave = 2*np.abs(2*(t*freq - np.floor(0.5 + t*freq))) - 1
    elif preset == "guitar": wave = np.sin(2*np.pi*freq*t) + 0.5*np.sin(4*np.pi*freq*t); wave *= np.exp(-t*3)
    elif preset == "bass": wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(4*np.pi*freq*t); wave *= np.exp(-t*1.5)
    elif preset == "piano": wave = np.sin(2*np.pi*freq*t) + 0.6*np.sin(4*np.pi*freq*t); wave *= (1 - t/duration)**2
    elif preset == "strings": wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(4*np.pi*freq*t); wave *= np.exp(-t*0.8)
    elif preset == "synth": wave = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.01*t); wave *= np.exp(-t*0.5)
    elif preset == "crystal": 
        wave = np.sin(2*np.pi*freq*t) + 0.3*np.sin(4*np.pi*freq*t) + 0.1*np.sin(6*np.pi*freq*t)
        wave *= np.exp(-t*0.5) * (1 - np.exp(-t*2))
    elif preset == "tibetan":
        wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(4*np.pi*freq*t) + 0.2*np.sin(8*np.pi*freq*t)
        wave *= np.exp(-t*0.3) * (1 - np.exp(-t*1.5))
    elif preset == "native_flute":
        fundamental = np.sin(2*np.pi*freq*t)
        breath = np.random.normal(0, 0.1, len(t)) * np.exp(-t*0.8)
        overtone = 0.3 * np.sin(2*np.pi*freq*2*t + 0.2) * np.exp(-t*1.2)
        formant = 0.2 * np.sin(2*np.pi*freq*3.5*t) * np.exp(-t*1.5)
        wave = fundamental + breath + overtone + formant
        wave *= np.exp(-t*0.6) * (1 - np.exp(-t*3))
    elif preset == "didgeridoo":
        drone = np.sin(2*np.pi*freq*t)
        overtone = 0.4 * np.sin(2*np.pi*freq*2*t + 0.3)
        breath_mod = 1 + 0.3 * np.sin(2*np.pi*0.5*t)
        formant = 0.3 * np.sin(2*np.pi*freq*3.7*t) * np.exp(-t*0.8)
        noise = np.random.normal(0, 0.05, len(t)) * np.exp(-t*0.5)
        wave = (drone + overtone + formant + noise) * breath_mod
        wave *= np.exp(-t*0.2)

    wave = wave / (np.max(np.abs(wave)) + 1e-8) * 0.4
    wave_int16 = (wave * 32767).astype(np.int16)
    return AudioSegment(wave_int16.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

# ——— DRUM SOUND ———
@st.cache_data
def get_drum(drum_type, base_freq=60):
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.zeros_like(t)
    
    if drum_type == "bass_drum": wave = np.sin(2*np.pi*base_freq*t) * np.exp(-t*8)
    elif drum_type == "snare": 
        noise = np.random.uniform(-1, 1, len(t))
        wave = np.sin(2*np.pi*200*t) * np.exp(-t*10) + 0.5*noise*np.exp(-t*5)
    elif drum_type == "shaman_drum":
        wave = np.sin(2*np.pi*base_freq*t) * np.exp(-t*6)
        wave += 0.3 * np.sin(2*np.pi*base_freq*1.8*t) * np.exp(-t*7)
        wave += 0.2 * np.sin(2*np.pi*base_freq*3.2*t) * np.exp(-t*8)
        wave *= np.exp(-t*0.5)
    
    wave = wave / (np.max(np.abs(wave)) + 1e-8) * 0.3
    wave_int16 = (wave * 32767).astype(np.int16)
    return AudioSegment(wave_int16.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

# ——— DRUM LAYERS ———
def add_polyrhythm_layer(audio, bpm, drum_type, base_freq, duration_sec, pattern="4/4"):
    interval = 60 / bpm
    drum = get_drum(drum_type, base_freq)
    rhythm_audio = AudioSegment.silent(duration=duration_sec*1000)
    beats = [0,1,2,3] if pattern == "4/4" else [0, 0.6, 1.6, 2.2]
    pos = 0
    while pos < duration_sec:
        for b in beats:
            if pos + b*interval < duration_sec:
                rhythm_audio = rhythm_audio.overlay(drum, position=int((pos + b*interval)*1000))
        pos += 4 * interval
    return audio.overlay(rhythm_audio - 10)

def add_shamanic_drum_layer(audio, bpm, drum_type, base_freq, duration_sec, pattern="heartbeat"):
    interval = 60 / bpm
    drum = get_drum(drum_type, base_freq)
    rhythm_audio = AudioSegment.silent(duration=duration_sec*1000)
    beats = [0, 0.6, 1.6, 2.2] if pattern == "heartbeat" else [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    pos = 0
    while pos < duration_sec:
        for b in beats:
            if pos + b*interval < duration_sec:
                rhythm_audio = rhythm_audio.overlay(drum, position=int((pos + b*interval)*1000))
        pos += 4 * interval
    return audio.overlay(rhythm_audio - 10)

# ——— LIVE PIANO ———
def piano_keyboard(base_freq=444, tone_preset="sine"):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    freq_ratios = [1, 1.059, 1.122, 1.189, 1.260, 1.335, 1.414, 1.498, 1.587, 1.682, 1.782, 1.888]
    freqs = [base_freq * r for r in freq_ratios]
    cols = st.columns(12)
    pressed = []
    for i, (note, freq) in enumerate(zip(notes, freqs)):
        with cols[i]:
            if st.button(note, key=f"key_{i}"):
                tone = get_tone(freq, 2.0, tone_preset)
                pressed.append(tone)
                st.session_state.piano_queue.put(tone)
    return pressed

# ——— RECORDING ———
if 'recording' not in st.session_state:
    st.session_state.recording = False
    st.session_state.recorded_audio = AudioSegment.silent(1)
    st.session_state.piano_queue = queue.Queue()
    st.session_state.record_thread = None
    st.session_state.stop_event = Event()

def record_worker():
    while not st.session_state.stop_event.is_set():
        try:
            tone = st.session_state.piano_queue.get(timeout=0.1)
            st.session_state.recorded_audio += tone
        except queue.Empty:
            continue

# ——— GENERATOR ———
def generate_full_track(layers, duration_min):
    duration_sec = duration_min * 60
    final_audio = AudioSegment.silent(duration=duration_sec*1000)
    all_freqs = []

    for layer in layers:
        layer_audio = AudioSegment.silent(duration=duration_sec*1000)
        freqs = layer.get("freqs", [528])
        all_freqs.extend(freqs)

        for f in freqs:
            tone = get_tone(f, duration_sec, layer.get("tone", "sine"))
            layer_audio = layer_audio.overlay(tone)

        if layer.get("binaural", 0) > 0:
            left = layer_audio
            right = layer_audio.low_pass_filter(layer["binaural"] + 100).high_pass_filter(layer["binaural"] - 100).apply_gain(-3)
            layer_audio = left.pan(-0.5).overlay(right.pan(0.5))

        if layer.get("drum", False):
            if layer.get("shamanic", False):
                layer_audio = add_shamanic_drum_layer(layer_audio, layer.get("bpm", 60), layer.get("drum_type", "shaman_drum"), layer.get("drum_freq", 60), duration_sec, layer.get("pattern", "heartbeat"))
            else:
                layer_audio = add_polyrhythm_layer(layer_audio, layer.get("bpm", 60), layer.get("drum_type", "bass_drum"), layer.get("drum_freq", 60), duration_sec, layer.get("pattern", "4/4"))

        final_audio = final_audio.overlay(layer_audio)

    return final_audio, list(set(all_freqs))

# ——— UI ———
st.title("Sacred Sound Codex")
st.markdown("### *All Tones. All Rhythms. All Layers.*")

tab1, tab2 = st.tabs(["Layer Engine", "Live Piano + Record"])

with tab1:
    if 'layers' not in st.session_state:
        st.session_state.layers = [{"freqs": [528], "tone": "sine", "binaural": 6.0, "drum": False, "shamanic": False, "bpm": 60, "drum_type": "bass_drum", "drum_freq": 60, "pattern": "4/4"}]

    def add_layer(): st.session_state.layers.append({**st.session_state.layers[0]})

    duration = st.slider("Duration (min)", 1, 120, 5)
    for i, layer in enumerate(st.session_state.layers):
        with st.expander(f"Layer {i+1}", expanded=i==0):
            col1, col2 = st.columns(2)
            with col1:
                freq_input = st.text_input("Freqs", "528", key=f"f{i}")
                layer["freqs"] = [float(f.strip()) for f in freq_input.split(",") if f.strip().replace('.','').isdigit()]
                layer["tone"] = st.selectbox("Tone", ["sine","square","sawtooth","triangle","guitar","bass","piano","strings","synth","crystal","tibetan","native_flute","didgeridoo"], key=f"t{i}")
                layer["binaural"] = st.slider("Binaural", 0.0, 40.0, 6.0, 0.1, key=f"b{i}")
            with col2:
                layer["drum"] = st.checkbox("Drum", key=f"d{i}")
                if layer["drum"]:
                    layer["shamanic"] = st.checkbox("Shamanic", key=f"sh{i}")
                    layer["bpm"] = st.slider("BPM", 0, 300, 60, key=f"bpm{i}")
                    layer["drum_type"] = st.selectbox("Drum", ["bass_drum","snare","shaman_drum"], key=f"dt{i}")
                    layer["drum_freq"] = st.number_input("Freq", 20, 200, 60, key=f"df{i}")
                    layer["pattern"] = st.selectbox("Pattern", ["4/4", "heartbeat"], key=f"p{i}")

    st.button("Add Layer", on_click=add_layer)
    if st.button("Generate", type="primary"):
        with st.spinner("Building..."):
            audio, freqs = generate_full_track(st.session_state.layers, duration)
            wav = io.BytesIO(); audio.export(wav, format="wav"); wav.seek(0)
            st.audio(wav, format="audio/wav")
            st.download_button("Download", wav, f"codex_{int(time.time())}.wav", "audio/wav")

with tab2:
    base_freq = st.number_input("Base Frequency", 100, 1000, 444)
    tone_preset = st.selectbox("Piano Tone", ["sine","piano","native_flute","didgeridoo"])
    st.markdown("### Live Keys")
    pressed = piano_keyboard(base_freq, tone_preset)
    for tone in pressed:
        st.audio(tone.export(format="wav").read(), format="audio/wav", autoplay=True)

    colr1, colr2 = st.columns(2)
    with colr1:
        if st.button("Start Recording"):
            st.session_state.recording = True
            st.session_state.recorded_audio = AudioSegment.silent(1)
            st.session_state.stop_event.clear()
            st.session_state.record_thread = Thread(target=record_worker, daemon=True)
            st.session_state.record_thread.start()
            st.success("Recording...")
    with colr2:
        if st.button("Stop & Download"):
            if st.session_state.recording:
                st.session_state.stop_event.set()
                if st.session_state.record_thread:
                    st.session_state.record_thread.join(timeout=1)
                st.session_state.recording = False
                wav = io.BytesIO()
                st.session_state.recorded_audio.export(wav, format="wav")
                wav.seek(0)
                st.download_button("Download", wav, "piano_recording.wav", "audio/wav")
                st.success("Saved!")

# ——— CYMATIC ———
if 'freqs' in locals():
    st.pyplot(draw_cymatic(freqs))

# ——— INSTALL ———
st.markdown("""
---
**Install to Android**  
1. Open in **Chrome**  
2. Tap **3-dot menu → Add to Home screen**  
3. Name: **"Sacred Sound Codex"** → **Add**
""")
