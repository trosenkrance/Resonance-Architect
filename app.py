# app.py — SACRED SOUND CODEX v8 — ON-SCREEN KEYBOARD + MP3 + NO ERRORS
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from threading import Thread, Event
import queue

# ——— PYDUB PATCH: NUMPY FALLBACK FOR audioop (Python 3.13) ———
import sys

class NumPyAudioop:
    def reverse(self, data, size): 
        return np.flip(np.frombuffer(data, dtype=np.int16)).tobytes()
    def lin2lin(self, data, size1, size2): 
        return data.astype(np.int16).tobytes()
    def ulaw2lin(self, data, size): return data
    def alaw2lin(self, data, size): return data
    def lin2ulaw(self, data, size): return data
    def lin2alaw(self, data, size): return data
    def lin2twos(self, data, size): return data
    def add(self, data1, data2, size):
        a1 = np.frombuffer(data1, dtype=np.int16)
        a2 = np.frombuffer(data2, dtype=np.int16)
        return (a1 + a2).clip(-32768, 32767).astype(np.int16).tobytes()
    def bias(self, data, size, bias):
        return (np.frombuffer(data, dtype=np.int16) + bias).clip(-32768, 32767).astype(np.int16).tobytes()
    def max(self, data, size): 
        return int(np.max(np.abs(np.frombuffer(data, dtype=np.int16))))
    def min(self, data, size): 
        return int(np.min(np.frombuffer(data, dtype=np.int16)))
    def mul(self, data, size, mul):
        return (np.frombuffer(data, dtype=np.int16) * mul).clip(-32768, 32767).astype(np.int16).tobytes()
    def ratecv(self, data, size, channels, in_rate, out_rate, state): return data
    def tomono(self, data, size, channels): return data
    def tostereo(self, data, size): return data
    def lin2adpcm(self, data, size, state): return data
    def adpcm2lin(self, data, size, state): return data

sys.modules['audioop'] = NumPyAudioop()
from pydub import AudioSegment

# ——— PWA + CONFIG ———
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

SAMPLE_RATE = 44100

# ——— TONE GENERATOR ———
@st.cache_data
def get_tone(freq, duration, preset="sine"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.zeros_like(t)
    
    if preset == "sine": 
        wave = np.sin(2*np.pi*freq*t)
    elif preset == "piano": 
        wave = np.sin(2*np.pi*freq*t) + 0.6*np.sin(4*np.pi*freq*t)
        wave *= (1 - t/duration)**2
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

    max_val = np.max(np.abs(wave)) + 1e-8
    wave = wave / max_val * 0.4
    wave_int16 = (wave * 32767).astype(np.int16)
    return AudioSegment(wave_int16.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

# ——— ON-SCREEN KEYBOARD ———
def on_screen_keyboard(base_freq=444, tone_preset="crystal"):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    midi_notes = list(range(48, 72))  # C3 to B4 (2 octaves)
    freqs = [base_freq * (2 ** ((m - 69) / 12.0)) for m in midi_notes]
    
    piano_css = """
    <style>
    .piano { display: flex; justify-content: center; margin: 20px 0; flex-wrap: wrap; }
    .white-key { 
        width: 40px; height: 160px; background: #fff; border: 1px solid #ccc; 
        margin: 0 1px; border-radius: 5px; display: flex; align-items: flex-end; 
        justify-content: center; font-weight: bold; color: #333; cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.1s;
    }
    .white-key:active { background: #00ff88; transform: translateY(2px); }
    .black-key { 
        width: 28px; height: 100px; background: #000; color: #fff; 
        margin: 0 4px 0 -14px; z-index: 2; border-radius: 0 0 5px 5px; 
        display: flex; align-items: flex-end; justify-content: center; font-size: 12px;
        cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: all 0.1s;
    }
    .black-key:active { background: #00ff88; transform: translateY(2px); }
    .key-label { margin-bottom: 10px; }
    @media (max-width: 600px) {
        .white-key { width: 32px; height: 140px; }
        .black-key { width: 22px; height: 90px; }
    }
    </style>
    """
    st.markdown(piano_css, unsafe_allow_html=True)

    piano_html = '<div class="piano">'
    black_positions = [1, 3, 6, 8, 10]  # C#, D#, F#, G#, A#
    
    for i in range(24):
        note_idx = i % 12
        is_black = note_idx in black_positions and i % 12 != 0
        freq = freqs[i]
        octave = 3 + (i // 12)
        note_name = notes[note_idx] + str(octave)
        
        key_class = "black-key" if is_black else "white-key"
        piano_html += f'''
        <div class="{key_class}" onclick="playNote({freq}, '{tone_preset}')">
            <div class="key-label">{note_name}</div>
        </div>
        '''
    piano_html += '</div>'

    js = f"""
    <script>
    function playNote(freq, preset) {{
        const url = `/?freq=${{freq}}&preset=${{preset}}&t=${{Date.now()}}`;
        window.location.href = url;
    }}
    </script>
    """
    st.markdown(piano_html + js, unsafe_allow_html=True)
    return freqs

# ——— TONE ENDPOINT (USING CACHE_RESOURCE) ———
@st.cache_resource
def _tone_cache():
    return {}

if "tone_cache" not in st.session_state:
    st.session_state.tone_cache = _tone_cache()

def get_cached_tone(freq, preset):
    key = f"{freq:.2f}_{preset}"
    cache = st.session_state.tone_cache
    if key not in cache:
        tone = get_tone(freq, 3.0, preset)
        wav = io.BytesIO()
        tone.export(wav, format="wav")
        cache[key] = wav.getvalue()
    return cache[key]

# Handle tone playback via URL
query_params = st.experimental_get_query_params()
if "freq" in query_params and "preset" in query_params:
    freq = float(query_params["freq"][0])
    preset = query_params["preset"][0]
    wav_data = get_cached_tone(freq, preset)
    st.audio(wav_data, format="audio/wav")
    st.stop()

# ——— RECORDING STATE ———
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

# ——— DRUM LAYER ———
def add_shamanic_drum_layer(audio, bpm, drum_type, base_freq, duration_sec, pattern="heartbeat"):
    interval = 60 / bpm
    drum = get_tone(base_freq, 0.5, "shaman_drum") if drum_type == "shaman_drum" else get_tone(base_freq, 0.5, "bass_drum")
    rhythm_audio = AudioSegment.silent(duration=duration_sec*1000)
    patterns = {"heartbeat": [0, 0.6, 1.6, 2.2], "4/4": [0, 1, 2, 3]}
    beats = patterns.get(pattern, [0, 1, 2, 3])
    pos = 0
    while pos < duration_sec:
        for b in beats:
            if pos + b*interval < duration_sec:
                rhythm_audio = rhythm_audio.overlay(drum, position=int((pos + b*interval)*1000))
        pos += 4 * interval
    return audio.overlay(rhythm_audio - 10)

# ——— MAIN GENERATOR ———
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
            beat = layer["binaural"]
            left = layer_audio
            right = layer_audio.low_pass_filter(beat + 100).high_pass_filter(beat - 100).apply_gain(-3)
            layer_audio = left.pan(-0.5).overlay(right.pan(0.5))

        if layer.get("drum", False):
            layer_audio = add_shamanic_drum_layer(
                layer_audio, layer.get("bpm", 60), layer.get("drum_type", "shaman_drum"),
                layer.get("drum_freq", 60), duration_sec, layer.get("pattern", "heartbeat")
            )

        final_audio = final_audio.overlay(layer_audio)

    return final_audio, list(set(all_freqs))

# ——— CYMATIC ———
def draw_cymatic(freqs, title="Cymatic Mandala"):
    fig, ax = plt.subplots(figsize=(7,7), facecolor='black')
    ax.set_facecolor('black')
    t = np.linspace(0, 2*np.pi, 1200)
    colors = ['#00ff88', '#ff00ff', '#00ffff', '#ffff00', '#ff8800', '#88ff00', '#ff0088']
    for i, f in enumerate(freqs[:7]):
        r = 1 + 0.4 * np.sin(12 * t + f * t * 0.08)
        x = r * np.cos(t * (f % 9))
        y = r * np.sin(t * (f % 9))
        ax.plot(x, y, color=colors[i % len(colors)], alpha=0.9, linewidth=1.8)
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=16, pad=20)
    return fig

# ——— UI ———
st.title("Sacred Sound Codex — ON-SCREEN KEYBOARD")
st.markdown("### *Tap. Play. Record. Ascend.*")

tab1, tab2 = st.tabs(["On-Screen Piano", "Layer Engine"])

with tab1:
    base_freq = st.slider("Base Frequency", 100, 1000, 444, key="base_freq_os")
    tone_preset = st.selectbox("Tone Preset", [
        "sine","piano","crystal","tibetan","native_flute","didgeridoo"
    ], key="tone_preset_os")

    st.markdown("### On-Screen Piano (2 Octaves)")
    freqs = on_screen_keyboard(base_freq, tone_preset)

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
                wav = io.BytesIO(); mp3 = io.BytesIO()
                st.session_state.recorded_audio.export(wav, format="wav")
                st.session_state.recorded_audio.export(mp3, format="mp3", bitrate="192k")
                wav.seek(0); mp3.seek(0)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("WAV", data=wav, file_name="piano_rec.wav", mime="audio/wav")
                with col2:
                    st.download_button("MP3", data=mp3, file_name="piano_rec.mp3", mime="audio/mpeg")
                st.success("Saved!")

with tab2:
    if 'layers' not in st.session_state:
        st.session_state.layers = [{
            "freqs": [528], "tone": "sine", "binaural": 6.0,
            "drum": False, "bpm": 60, "drum_type": "shaman_drum", "drum_freq": 60, "pattern": "heartbeat"
        }]

    def add_layer(): 
        st.session_state.layers.append({**st.session_state.layers[0]})

    duration = st.slider("Duration (min)", 1, 120, 5, key="duration")
    
    for i, layer in enumerate(st.session_state.layers):
        with st.expander(f"Layer {i+1}", expanded=(i==0)):
            col1, col2 = st.columns(2)
            with col1:
                freq_input = st.text_input("Freqs", "528", key=f"f{i}")
                layer["freqs"] = [float(f.strip()) for f in freq_input.split(",") if f.strip().replace('.','').isdigit()]
                layer["tone"] = st.selectbox("Tone", [
                    "sine","square","sawtooth","triangle","guitar","bass","piano","strings",
                    "synth","crystal","tibetan","native_flute","didgeridoo"
                ], key=f"t{i}")
                layer["binaural"] = st.slider("Binaural", 0.0, 40.0, 6.0, 0.1, key=f"b{i}")
            with col2:
                layer["drum"] = st.checkbox("Drum", key=f"d{i}")
                if layer["drum"]:
                    layer["bpm"] = st.slider("BPM", 30, 300, 60, key=f"bpm{i}")
                    layer["drum_type"] = st.selectbox("Drum", ["bass_drum","snare","shaman_drum"], key=f"dt{i}")
                    layer["drum_freq"] = st.number_input("Freq", 20, 200, 60, key=f"df{i}")
                    layer["pattern"] = st.selectbox("Pattern", ["heartbeat", "4/4"], key=f"p{i}")

    st.button("Add Layer", on_click=add_layer)
    
    if st.button("Generate Track", type="primary"):
        with st.spinner("Building..."):
            audio, freqs = generate_full_track(st.session_state.layers, duration)
            wav = io.BytesIO(); mp3 = io.BytesIO()
            audio.export(wav, format="wav"); audio.export(mp3, format="mp3", bitrate="192k")
            wav.seek(0); mp3.seek(0)
            st.audio(wav, format="audio/wav")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("WAV", data=wav, file_name=f"codex_{int(time.time())}.wav", mime="audio/wav")
            with col2:
                st.download_button("MP3", data=mp3, file_name=f"codex_{int(time.time())}.mp3", mime="audio/mpeg")

# ——— CYMATIC DISPLAY ———
if 'freqs' in locals():
    st.pyplot(draw_cymatic(freqs))

# ——— INSTALL GUIDE ———
st.markdown("""
---
**Install to Android**  
1. Open in **Chrome**  
2. Tap **3-dot menu → Add to Home screen**  
3. Name: **"Sacred Sound Codex"** → **Add**
""")
