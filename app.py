# app.py — SACRED SOUND CODEX v12 — BINAURAL PRESETS + 28 TONES + FULL FEATURES
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from threading import Thread, Event
import queue
import base64

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

# ——— BINAURAL PRESETS ———
BINAURAL_PRESETS = {
    "None": 0.0,
    "Delta (Deep Sleep)": 2.0,
    "Theta (Meditation)": 6.0,
    "Alpha (Relaxed Focus)": 10.0,
    "Beta (Alert)": 18.0,
    "Gamma (Insight)": 40.0,
    "HyperGamma (Neural Sync)": 100.0,
    "Schumann Resonance": 7.83,
    "40Hz Neural Boost": 40.0
}

# ——— 28 SACRED TONE PRESETS ———
@st.cache_data
def get_tone(freq, duration=3.0, preset="sine"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.zeros_like(t)
    
    # ——— CLASSIC WAVEFORMS ———
    if preset == "sine": 
        wave = np.sin(2*np.pi*freq*t)
    elif preset == "square": 
        wave = np.sign(np.sin(2*np.pi*freq*t))
    elif preset == "sawtooth": 
        wave = 2*(t*freq - np.floor(0.5 + t*freq))
    elif preset == "triangle": 
        wave = 2*np.abs(2*(t*freq - np.floor(0.5 + t*freq))) - 1

    # ——— INSTRUMENTS ———
    elif preset == "guitar": 
        wave = np.sin(2*np.pi*freq*t) + 0.5*np.sin(4*np.pi*freq*t)
        wave *= np.exp(-t*3)
    elif preset == "bass": 
        wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(4*np.pi*freq*t)
        wave *= np.exp(-t*1.5)
    elif preset == "piano": 
        wave = np.sin(2*np.pi*freq*t) + 0.6*np.sin(4*np.pi*freq*t)
        wave *= (1 - t/duration)**2
    elif preset == "strings": 
        wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(4*np.pi*freq*t)
        wave *= np.exp(-t*0.8)
    elif preset == "synth": 
        wave = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.01*t)
        wave *= np.exp(-t*0.5)

    # ——— CRYSTAL BOWLS & BELLS ———
    elif preset == "crystal": 
        wave = np.sin(2*np.pi*freq*t) + 0.3*np.sin(4*np.pi*freq*t) + 0.1*np.sin(6*np.pi*freq*t)
        wave *= np.exp(-t*0.5) * (1 - np.exp(-t*2))
    elif preset == "tibetan": 
        wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(4*np.pi*freq*t) + 0.2*np.sin(8*np.pi*freq*t)
        wave *= np.exp(-t*0.3) * (1 - np.exp(-t*1.5))
    elif preset == "singing_bowl": 
        wave = np.sin(2*np.pi*freq*t) + 0.6*np.sin(2*np.pi*freq*2.1*t) + 0.3*np.sin(2*np.pi*freq*3.3*t)
        wave *= np.exp(-t*0.4) * (1 - np.exp(-t*1.8))
    elif preset == "quartz_bell": 
        wave = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.8*t) + 0.2*np.sin(2*np.pi*freq*3.9*t)
        wave *= np.exp(-t*0.6) * (1 + 0.3*np.sin(2*np.pi*3*t))

    # ——— WOODWINDS & BREATH ———
    elif preset == "native_flute": 
        fundamental = np.sin(2*np.pi*freq*t)
        breath = np.random.normal(0, 0.1, len(t)) * np.exp(-t*0.8)
        overtone = 0.3 * np.sin(2*np.pi*freq*2*t + 0.2) * np.exp(-t*1.2)
        formant = 0.2 * np.sin(2*np.pi*freq*3.5*t) * np.exp(-t*1.5)
        wave = fundamental + breath + overtone + formant
        wave *= np.exp(-t*0.6) * (1 - np.exp(-t*3))
    elif preset == "pan_flute": 
        wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(2*np.pi*freq*2*t) + 0.1*np.sin(2*np.pi*freq*4*t)
        wave *= np.exp(-t*0.7) * (1 - np.exp(-t*2.5))
    elif preset == "shakuhachi": 
        wave = np.sin(2*np.pi*freq*t) + 0.3*np.sin(2*np.pi*freq*2*t + 0.5)
        wave *= np.exp(-t*0.9) * (1 - np.exp(-t*2))

    # ——— DIDGERIDOO & DRONES ———
    elif preset == "didgeridoo": 
        drone = np.sin(2*np.pi*freq*t)
        overtone = 0.4 * np.sin(2*np.pi*freq*2*t + 0.3)
        breath_mod = 1 + 0.3 * np.sin(2*np.pi*0.5*t)
        formant = 0.3 * np.sin(2*np.pi*freq*3.7*t) * np.exp(-t*0.8)
        noise = np.random.normal(0, 0.05, len(t)) * np.exp(-t*0.5)
        wave = (drone + overtone + formant + noise) * breath_mod
        wave *= np.exp(-t*0.2)
    elif preset == "overtone_drone": 
        wave = np.sin(2*np.pi*freq*t) + 0.8*np.sin(2*np.pi*freq*2*t) + 0.5*np.sin(2*np.pi*freq*3*t)
        wave *= np.exp(-t*0.1)

    # ——— PLANETARY FREQUENCIES (Hans Cousto) ———
    elif preset == "earth_year":      # 136.1 Hz
        wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(2*np.pi*freq*2*t) + 0.2*np.sin(2*np.pi*freq*3*t)
        wave *= np.exp(-t*0.3)
    elif preset == "moon_synodic":    # 210.42 Hz
        wave = np.sin(2*np.pi*freq*t) + 0.3*np.sin(2*np.pi*freq*2.1*t)
        wave *= np.exp(-t*0.4)
    elif preset == "sun":             # 126.22 Hz
        wave = np.sin(2*np.pi*freq*t) + 0.6*np.sin(2*np.pi*freq*2*t) + 0.3*np.sin(2*np.pi*freq*3*t)
        wave *= np.exp(-t*0.2)
    elif preset == "mercury":         # 141.27 Hz
        wave = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.9*t)
        wave *= np.exp(-t*0.5)

    # ——— SOLFEGGIO & ANCIENT ———
    elif preset == "solfeggio_528": 
        wave = np.sin(2*np.pi*freq*t) + 0.4*np.sin(2*np.pi*freq*2*t) + 0.1*np.sin(2*np.pi*freq*3*t)
        wave *= np.exp(-t*0.4) * (1 - np.exp(-t*1.5))
    elif preset == "pythagorean_a": 
        wave = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.5*t)  # Just intonation 3:2
        wave *= np.exp(-t*0.6)
    elif preset == "om_chant": 
        wave = np.sin(2*np.pi*freq*t) + 0.3*np.sin(2*np.pi*freq*2*t) + 0.1*np.sin(2*np.pi*freq*3*t)
        wave *= np.exp(-t*0.3) * (1 + 0.2*np.sin(2*np.pi*1.5*t))

    # ——— SHAMANIC & NATURE ———
    elif preset == "rainstick": 
        noise = np.random.normal(0, 0.3, len(t))
        wave = noise * np.exp(-t*5) * (1 + 0.5*np.sin(2*np.pi*10*t))
    elif preset == "ocean_wave": 
        wave = np.sin(2*np.pi*0.2*t) * np.sin(2*np.pi*freq*t)
        wave *= np.exp(-t*0.1)

    max_val = np.max(np.abs(wave)) + 1e-8
    wave = wave / max_val * 0.4
    wave_int16 = (wave * 32767).astype(np.int16)
    return AudioSegment(wave_int16.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

# ——— DRUM ENGINE ———
@st.cache_data
def get_drum(drum_type, base_freq=60):
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.zeros_like(t)
    
    if drum_type == "bass_drum": 
        wave = np.sin(2*np.pi*base_freq*t) * np.exp(-t*8)
    elif drum_type == "snare": 
        noise = np.random.uniform(-1, 1, len(t))
        wave = np.sin(2*np.pi*200*t) * np.exp(-t*10) + 0.5*noise*np.exp(-t*5)
    elif drum_type == "shaman_drum":
        wave = np.sin(2*np.pi*base_freq*t) * np.exp(-t*6)
        wave += 0.3 * np.sin(2*np.pi*base_freq*1.8*t) * np.exp(-t*7)
        wave += 0.2 * np.sin(2*np.pi*base_freq*3.2*t) * np.exp(-t*8)
        wave *= np.exp(-t*0.5)
    
    max_val = np.max(np.abs(wave)) + 1e-8
    wave = wave / max_val * 0.3
    wave_int16 = (wave * 32767).astype(np.int16)
    return AudioSegment(wave_int16.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

# ——— DRUM LAYERS ———
def add_shamanic_drum_layer(audio, bpm, drum_type, base_freq, duration_sec, pattern="heartbeat"):
    interval = 60 / bpm
    drum = get_drum(drum_type, base_freq)
    rhythm_audio = AudioSegment.silent(duration=duration_sec*1000)
    
    patterns = {
        "heartbeat": [0, 0.6, 1.6, 2.2],
        "trance": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "journey": [0, 1.0, 2.5, 3.0, 4.0, 5.0],
        "4/4": [0, 1, 2, 3]
    }
    beats = patterns.get(pattern, [0, 1, 2, 3])
    
    pos = 0
    while pos < duration_sec:
        for b in beats:
            if pos + b*interval < duration_sec:
                position_ms = int((pos + b*interval) * 1000)
                rhythm_audio = rhythm_audio.overlay(drum, position=position_ms)
        pos += 4 * interval
    return audio.overlay(rhythm_audio - 10)

# ——— ON-SCREEN KEYBOARD ———
def render_piano(base_freq, preset, binaural_beat=0.0):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    midi_start = 48  # C3
    html = """
    <style>
    .piano-container { display: flex; justify-content: center; margin: 20px 0; flex-wrap: nowrap; }
    .key { width: 40px; height: 160px; margin: 0 1px; border-radius: 5px; 
        display: flex; align-items: flex-end; justify-content: center; 
        font-weight: bold; cursor: pointer; transition: all 0.1s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .white { background: #fff; color: #333; border: 1px solid #ccc; }
    .black { width: 28px; height: 100px; background: #000; color: #fff; 
        margin: 0 4px 0 -14px; z-index: 2; border-radius: 0 0 5px 5px; 
        font-size: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .key:active { background: #00ff88 !important; transform: translateY(2px); }
    .label { margin-bottom: 10px; }
    @media (max-width: 600px) {
        .key { width: 32px; height: 140px; }
        .black { width: 22px; height: 90px; }
    }
    </style>
    <div class="piano-container">
    """
    
    for i in range(24):
        midi = midi_start + i
        freq = base_freq * (2 ** ((midi - 69) / 12.0))
        note_idx = i % 12
        is_black = note_idx in [1, 3, 6, 8, 10]
        octave = 3 + (i // 12)
        label = notes[note_idx] + str(octave)
        key_class = "black" if is_black else "white"
        
        tone = get_tone(freq, 3.0, preset)
        if binaural_beat > 0:
            left = tone.pan(-0.5)
            right = get_tone(freq + binaural_beat, 3.0, preset).pan(0.5)
            tone = left.overlay(right)
        
        wav = io.BytesIO()
        tone.export(wav, format="wav")
        b64 = base64.b64encode(wav.getvalue()).decode()
        audio_url = f"data:audio/wav;base64,{b64}"
        
        html += f'''
        <div class="key {key_class}" onclick="playSound('{audio_url}')">
            <div class="label">{label}</div>
        </div>
        '''
    
    html += """
    </div>
    <script>
    function playSound(url) {
        const audio = new Audio(url);
        audio.play().catch(() => {});
    }
    </script>
    """
    return html

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

        binaural_beat = layer.get("binaural", 0.0)
        if binaural_beat > 0:
            left = layer_audio
            right = layer_audio.low_pass_filter(binaural_beat + 100).high_pass_filter(binaural_beat - 100).apply_gain(-3)
            layer_audio = left.pan(-0.5).overlay(right.pan(0.5))

        if layer.get("drum", False):
            layer_audio = add_shamanic_drum_layer(
                layer_audio,
                layer.get("bpm", 60),
                layer.get("drum_type", "shaman_drum"),
                layer.get("drum_freq", 60),
                duration_sec,
                layer.get("pattern", "heartbeat")
            )

        final_audio = final_audio.overlay(layer_audio)

    return final_audio, list(set(all_freqs))

# ——— CYMATIC MANDALA ———
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
st.title("Sacred Sound Codex — BINAURAL + 28 TONES")
st.markdown("### *Delta • Theta • Gamma • 40Hz • Schumann • Crystal • Didgeridoo*")

tab1, tab2 = st.tabs(["On-Screen Piano", "Layer Engine"])

# ——— TAB 1: ON-SCREEN PIANO ———
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        base_freq = st.slider("Base Frequency (Hz)", 100, 1000, 444, key="base_freq_piano")
    with col2:
        binaural_preset = st.selectbox("Binaural Preset", list(BINAURAL_PRESETS.keys()), key="binaural_preset_piano")
    
    preset = st.selectbox("Tone Preset", [
        "sine","square","sawtooth","triangle",
        "guitar","bass","piano","strings","synth",
        "crystal","tibetan","singing_bowl","quartz_bell",
        "native_flute","pan_flute","shakuhachi",
        "didgeridoo","overtone_drone",
        "earth_year","moon_synodic","sun","mercury",
        "solfeggio_528","pythagorean_a","om_chant",
        "rainstick","ocean_wave"
    ], key="preset_piano")

    st.markdown("### On-Screen Piano (C3–B4)")
    piano_html = render_piano(base_freq, preset, BINAURAL_PRESETS[binaural_preset])
    st.components.v1.html(piano_html, height=220, scrolling=False)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording", key="start_rec"):
            st.session_state.recording = True
            st.session_state.recorded_audio = AudioSegment.silent(1)
            st.session_state.stop_event.clear()
            st.session_state.record_thread = Thread(target=record_worker, daemon=True)
            st.session_state.record_thread.start()
            st.success("Recording...")
    with col2:
        if st.button("Stop & Download", key="stop_rec"):
            if st.session_state.recording:
                st.session_state.stop_event.set()
                if st.session_state.record_thread:
                    st.session_state.record_thread.join(timeout=1)
                st.session_state.recording = False
                wav = io.BytesIO(); mp3 = io.BytesIO()
                st.session_state.recorded_audio.export(wav, format="wav")
                st.session_state.recorded_audio.export(mp3, format="mp3", bitrate="192k")
                wav.seek(0); mp3.seek(0)
                c1, c2 = st.columns(2)
                with c1: st.download_button("WAV", data=wav, file_name="piano_rec.wav", mime="audio/wav")
                with c2: st.download_button("MP3", data=mp3, file_name="piano_rec.mp3", mime="audio/mpeg")
                st.success("Saved!")

# ——— TAB 2: LAYER ENGINE ———
with tab2:
    if 'layers' not in st.session_state:
        st.session_state.layers = [{
            "freqs": [528], "tone": "crystal", "binaural_preset": "Theta (Meditation)", "binaural": 6.0,
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
                    "sine","square","sawtooth","triangle",
                    "guitar","bass","piano","strings","synth",
                    "crystal","tibetan","singing_bowl","quartz_bell",
                    "native_flute","pan_flute","shakuhachi",
                    "didgeridoo","overtone_drone",
                    "earth_year","moon_synodic","sun","mercury",
                    "solfeggio_528","pythagorean_a","om_chant",
                    "rainstick","ocean_wave"
                ], key=f"t{i}")
                binaural_preset = st.selectbox("Binaural Preset", list(BINAURAL_PRESETS.keys()), key=f"bp{i}")
                layer["binaural_preset"] = binaural_preset
                layer["binaural"] = BINAURAL_PRESETS[binaural_preset]
            with col2:
                layer["drum"] = st.checkbox("Drum", key=f"d{i}")
                if layer["drum"]:
                    layer["bpm"] = st.slider("BPM", 30, 300, 60, key=f"bpm{i}")
                    layer["drum_type"] = st.selectbox("Drum", ["bass_drum","snare","shaman_drum"], key=f"dt{i}")
                    layer["drum_freq"] = st.number_input("Freq", 20, 200, 60, key=f"df{i}")
                    layer["pattern"] = st.selectbox("Pattern", ["heartbeat", "trance", "journey", "4/4"], key=f"p{i}")

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
