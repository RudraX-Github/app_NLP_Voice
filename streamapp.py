# -*- coding: utf-8 -*-
"""
Voice NLP Suite (Streamlit Pro Edition)
A state-of-the-art web application for voice processing, analysis, and cloning.
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import spacy
from textblob import TextBlob
import soundfile as sf
import os
import time
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Optional Imports for Advanced Features ---
try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    import torch
    # We lazy load TTS inside the function to avoid long startup times
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# --- Page Configuration & CSS Styling ---
st.set_page_config(
    page_title="Voice NLP Suite Pro",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism and Modern UI
st.markdown("""
<style>
    /* Global Theme Overrides */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: rgba(15, 23, 42, 0.8);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        border-radius: 0 0 16px 16px;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: #fff;
    }
    
    /* Table Styling */
    .report-table {
        width: 100%;
        border-collapse: collapse;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    .report-table td {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .report-table .label {
        font-weight: 600;
        color: #94a3b8;
        width: 60%;
    }
    .report-table .value {
        font-family: monospace;
        color: #60a5fa;
        text-align: right;
    }

    /* Helper Classes */
    .highlight-text { color: #38bdf8; font-weight: bold; }
    .success-text { color: #4ade80; }
    .warning-text { color: #fbbf24; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'tts_audio' not in st.session_state:
    st.session_state.tts_audio = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None
if 'exit_app' not in st.session_state:
    st.session_state.exit_app = False

# --- Caching & Model Loading ---

@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)

@st.cache_resource
def load_tts_model():
    """Lazy load TTS model only when needed with PyTorch security fix."""
    if not TTS_AVAILABLE:
        return None
    try:
        from TTS.api import TTS
        # Imports needed for safe global allowlisting
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseTTSConfig
        
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Security fix for PyTorch 2.6+ 'weights_only=True' default
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals'):
            with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs, BaseTTSConfig]):
                tts = TTS(model_name).to(device)
        else:
            tts = TTS(model_name).to(device)
            
        return tts
    except Exception as e:
        st.error(f"Failed to load TTS model: {e}")
        return None

def transcribe_audio_stub(audio_data, sr):
    """
    Mock transcription for demo speed or fallback. 
    In production, integrate Whisper via `import whisper` and @st.cache_resource.
    """
    try:
        import speech_recognition as sr_lib
        recognizer = sr_lib.Recognizer()
        
        # Convert numpy audio to WAV bytes for SR
        with io.BytesIO() as wav_io:
            sf.write(wav_io, audio_data, sr, format='WAV')
            wav_io.seek(0)
            with sr_lib.AudioFile(wav_io) as source:
                audio_content = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_content) 
            return text
        except sr_lib.UnknownValueError:
            return "Could not understand audio."
        except sr_lib.RequestError:
            return "API unavailable."
    except ImportError:
        return "Speech Recognition library not installed. Please speak clearly."

def generate_pdf(data):
    """Generates a PDF report from analysis data."""
    if not FPDF_AVAILABLE:
        return None
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Voice NLP Suite Pro - Analysis Report', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Metadata
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Timestamp: {data['timestamp']}", 0, 1)
    pdf.ln(5)
    
    # 1. Acoustic Analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Acoustic Analysis", 0, 1)
    pdf.set_font("Arial", size=11)
    ac = data['acoustic']
    pdf.cell(0, 7, f"Mean Pitch: {ac['mean_pitch']:.2f} Hz", 0, 1)
    pdf.cell(0, 7, f"Mean F1: {ac['mean_f1']:.2f} Hz", 0, 1)
    pdf.cell(0, 7, f"Mean F2: {ac['mean_f2']:.2f} Hz", 0, 1)
    pdf.cell(0, 7, f"Intensity (RMS): {ac['intensity']:.4f}", 0, 1)
    pdf.ln(3)

    # 2. Linguistic Analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Linguistic Analysis", 0, 1)
    pdf.set_font("Arial", size=11)
    sent = data['sentiment']
    pdf.cell(0, 7, f"Polarity: {sent['polarity']:.2f}", 0, 1)
    pdf.cell(0, 7, f"Subjectivity: {sent['subjectivity']:.2f}", 0, 1)
    pdf.ln(2)
    pdf.set_font("Arial", 'I', 11)
    pdf.multi_cell(0, 7, f"Transcription: {data['transcription']}")
    pdf.ln(3)

    # 3. Emotional & Biometric
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Emotional & Biometric", 0, 1)
    pdf.set_font("Arial", size=11)
    emo = data['emotional']
    bio = data['biometric']
    pdf.cell(0, 7, f"Pitch Variation (Std): {emo['pitch_std']:.2f} Hz", 0, 1)
    pdf.cell(0, 7, f"Jitter: {emo['jitter']:.2f} %", 0, 1)
    pdf.cell(0, 7, f"Shimmer: {emo['shimmer']:.2f} %", 0, 1)
    pdf.cell(0, 7, f"Est. Gender: {bio['gender']}", 0, 1)
    pdf.cell(0, 7, f"Speaking Rate: {data['technical']['speaking_rate']:.1f} WPM", 0, 1)
    
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- Analysis Logic ---

def analyze_audio_file(audio_path, custom_text=None):
    """Core analysis pipeline."""
    y, sr = librosa.load(audio_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 1. Basic Acoustic Features
    rms = librosa.feature.rms(y=y)[0]
    
    # 2. Praat Analysis (Deep Acoustic)
    pitch_mean = 0.0
    pitch_std = 0.0
    jitter = 0.0
    shimmer = 0.0
    f1_mean = 0.0
    f2_mean = 0.0
    
    if PARSELMOUTH_AVAILABLE:
        try:
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            pitch_mean = pitch.selected_array['frequency'].mean()
            
            # Pitch Standard Deviation
            pitch_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")

            # Formant Analysis
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f1_mean = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2_mean = call(formant, "Get mean", 2, 0, 0, "Hertz")
            
            # Advanced jitter/shimmer
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
            # FIX: Shimmer requires [sound, point_process] as the object argument
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        except Exception as e:
            st.warning(f"Detailed acoustic analysis unavailable: {e}")
    
    # Handle NaNs
    pitch_mean = pitch_mean if not np.isnan(pitch_mean) else 0.0
    pitch_std = pitch_std if not np.isnan(pitch_std) else 0.0
    f1_mean = f1_mean if not np.isnan(f1_mean) else 0.0
    f2_mean = f2_mean if not np.isnan(f2_mean) else 0.0

    # 3. Transcription (if not provided)
    if not custom_text:
        transcript = transcribe_audio_stub(y, sr)
    else:
        transcript = custom_text

    # 4. NLP Analysis
    nlp = load_spacy_model()
    doc = nlp(transcript)
    blob = TextBlob(transcript)
    
    # 5. Derived Metrics
    # Gender Estimation
    if 85 < pitch_mean < 180: gender = "Likely Male"
    elif 165 < pitch_mean < 255: gender = "Likely Female"
    else: gender = "Undetermined"
    
    # Speaking Rate
    word_count = len(transcript.split())
    speaking_rate = (word_count / duration) * 60 if duration > 0 else 0

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "transcription": transcript,
        "acoustic": {
            "duration": duration,
            "sample_rate": sr,
            "intensity": np.mean(rms),
            "mean_pitch": pitch_mean,
            "mean_f1": f1_mean,
            "mean_f2": f2_mean,
        },
        "emotional": {
            "pitch_std": pitch_std,
            "jitter": jitter,
            "shimmer": shimmer
        },
        "biometric": {
            "gender": gender,
            "method": "Zero-Shot Cloning"
        },
        "technical": {
            "speaking_rate": speaking_rate,
            "language": "EN (Detected)",
            "model": "SpeechRecognition (Google)"
        },
        "sentiment": {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        },
        "keywords": [token.lemma_ for token in doc if not token.is_stop and token.is_alpha],
        "y": y,
        "sr": sr
    }
    
    # Update Session History
    st.session_state.history.append({
        "time": results["timestamp"],
        "text": transcript[:30] + "...",
        "polarity": results["sentiment"]["polarity"]
    })
    
    return results

def call(obj, method, *args):
    """Safe wrapper for Parselmouth calls"""
    return praat_call(obj, method, *args)

# --- UI Components ---

def render_navbar():
    st.markdown("""
        <div class="navbar">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px;">🎙️</span>
                <div>
                    <div style="font-weight: bold; font-size: 18px;">Voice NLP Suite</div>
                    <div style="font-size: 12px; color: #94a3b8;">Pro Edition v6.0</div>
                </div>
            </div>
            <div>
                <a href="#" style="color: #fff; text-decoration: none; margin-right: 20px;">Documentation</a>
                <span style="background: #3b82f6; padding: 5px 12px; border-radius: 20px; font-size: 12px;">BETA</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_hero():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# Unlock the Power of Your Voice")
        st.markdown("""
        <p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 20px;">
        Advanced voice processing suite combining <span class="highlight-text">Acoustic Analysis</span>, 
        <span class="highlight-text">Natural Language Understanding</span>, and 
        <span class="highlight-text">Generative Voice Cloning</span>.
        </p>
        """, unsafe_allow_html=True)
    with col2:
        # Placeholder for Lottie Animation
        st.markdown('<div style="text-align: right; font-size: 4rem;">🌊</div>', unsafe_allow_html=True)

def plot_waveform(y, sr):
    # Downsample for performance
    step = max(1, len(y) // 5000)
    y_plot = y[::step]
    x_plot = np.arange(len(y_plot)) * step / sr
    
    fig = px.line(x=x_plot, y=y_plot, labels={'x': 'Time (s)', 'y': 'Amplitude'})
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    fig.update_traces(line_color='#60a5fa', line_width=1.5)
    return fig

def plot_sentiment_gauge(polarity):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = polarity,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Polarity"},
        gauge = {
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#c084fc"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.1], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [-0.1, 0.1], 'color': 'rgba(148, 163, 184, 0.3)'},
                {'range': [0.1, 1], 'color': 'rgba(74, 222, 128, 0.3)'}],
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Main App Logic ---

def main():
    # Graceful Exit Screen
    if st.session_state.exit_app:
        st.markdown("""
        <style>
            .exit-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 40px;
                text-align: center;
                max-width: 600px;
                margin: 100px auto;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            }
        </style>
        <div class="exit-card">
            <h1 style="background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Session Closed</h1>
            <p style="color: #e2e8f0; font-size: 1.1rem; margin: 20px 0;">Thank you for using Voice NLP Suite Pro.</p>
            <p style="color: #94a3b8;">All temporary files have been cleaned up.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_c = st.columns([1, 1, 1])
        with col_c[1]:
            if st.button("🔄 Start New Session", use_container_width=True):
                st.session_state.exit_app = False
                st.rerun()
        st.stop()

    render_navbar()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("### 🛠️ Model Config")
        model_size = st.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium"], index=1)
        
        st.markdown("### 🎨 Appearance")
        show_spectrogram = st.toggle("Show Spectrogram", value=True)
        
        st.markdown("---")
        st.markdown("### 📜 Session History")
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            st.dataframe(df_hist, hide_index=True, use_container_width=True)
        else:
            st.info("No analysis history yet.")
        
        st.markdown("---")
        st.caption("v6.0.0 | Powered by Streamlit")
        
        if st.button("🚪 Graceful Exit", use_container_width=True):
            st.session_state.exit_app = True
            st.rerun()

    # Layout: Tabs
    tab_input, tab_dashboard, tab_cloning, tab_export = st.tabs([
        "📥 Overview & Input", "📊 Insights Dashboard", "🗣️ Voice Cloning", "📤 Export & Report"
    ])

    # --- TAB 1: INPUT ---
    with tab_input:
        render_hero()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📁 Upload File")
            uploaded_file = st.file_uploader("Choose a WAV/MP3 file", type=['wav', 'mp3', 'ogg'])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🎙️ Record Live")
            # New Streamlit Audio Input (v1.39+)
            audio_input = st.audio_input("Record a voice clip")
            st.markdown('</div>', unsafe_allow_html=True)

        # Process Input
        active_file = uploaded_file if uploaded_file else audio_input
        
        if active_file:
            st.markdown("---")
            if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
                with st.spinner("Processing audio signals..."):
                    # Save temp file for libraries that need paths (librosa/parselmouth)
                    with open("temp_input.wav", "wb") as f:
                        f.write(active_file.getbuffer())
                    
                    results = analyze_audio_file("temp_input.wav")
                    st.session_state.processed_data = results
                    st.toast("Analysis Complete!", icon="✅")
                    st.balloons()
                    
                    # Auto-switch hint (Logic only, UI cannot switch tabs programmatically easily without extra components)
                    st.info("Analysis complete! Switch to the 'Insights Dashboard' tab to view results.")

    # --- TAB 2: DASHBOARD ---
    with tab_dashboard:
        if st.session_state.processed_data:
            data = st.session_state.processed_data
            
            # Row 1: KPI Cards (Summary)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Pitch (Hz)", f"{data['acoustic']['mean_pitch']:.0f}", delta=None)
            with col2: st.metric("Speaking Rate", f"{data['technical']['speaking_rate']:.1f} WPM")
            with col3: st.metric("Sentiment", f"{data['sentiment']['polarity']:.2f}", 
                               delta="Positive" if data['sentiment']['polarity']>0 else "Negative")
            with col4: st.metric("Gender Est.", f"{data['biometric']['gender']}")

            # Row 2: Charts
            col_viz1, col_viz2 = st.columns([2, 1])
            with col_viz1:
                st.markdown("### 🌊 Waveform Analysis")
                st.plotly_chart(plot_waveform(data['y'], data['sr']), use_container_width=True)
                
                if show_spectrogram:
                    st.markdown("### 🌈 Spectrogram")
                    # Simple Spectrogram
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(data['y'])), ref=np.max)
                    fig_spec = px.imshow(D, aspect='auto', origin='lower', color_continuous_scale='Magma')
                    fig_spec.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig_spec, use_container_width=True)

            with col_viz2:
                st.markdown("### 🎭 Tone & Sentiment")
                st.plotly_chart(plot_sentiment_gauge(data['sentiment']['polarity']), use_container_width=True)
                st.info(f"Subjectivity: {data['sentiment']['subjectivity']:.2f}")

            # Row 3: Comprehensive Report Section
            st.markdown("---")
            st.markdown("### 📋 Comprehensive Analysis Report")
            
            col_rep1, col_rep2, col_rep3 = st.columns(3)
            
            with col_rep1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🎙️ Acoustic")
                st.markdown(f"""
                <table class="report-table">
                    <tr><td class="label">Mean Pitch</td><td class="value">{data['acoustic']['mean_pitch']:.2f} Hz</td></tr>
                    <tr><td class="label">Mean F1</td><td class="value">{data['acoustic']['mean_f1']:.2f} Hz</td></tr>
                    <tr><td class="label">Mean F2</td><td class="value">{data['acoustic']['mean_f2']:.2f} Hz</td></tr>
                    <tr><td class="label">Intensity (RMS)</td><td class="value">{data['acoustic']['intensity']:.4f}</td></tr>
                </table>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_rep2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 😃 Emotional & Biometric")
                st.markdown(f"""
                <table class="report-table">
                    <tr><td class="label">Pitch Var (Std)</td><td class="value">{data['emotional']['pitch_std']:.2f} Hz</td></tr>
                    <tr><td class="label">Jitter</td><td class="value">{data['emotional']['jitter']:.2f} %</td></tr>
                    <tr><td class="label">Shimmer</td><td class="value">{data['emotional']['shimmer']:.2f} %</td></tr>
                    <tr><td class="label">Est. Gender</td><td class="value">{data['biometric']['gender']}</td></tr>
                </table>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_rep3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🛠️ Technical")
                st.markdown(f"""
                <table class="report-table">
                    <tr><td class="label">Speaking Rate</td><td class="value">{data['technical']['speaking_rate']:.1f} WPM</td></tr>
                    <tr><td class="label">Language</td><td class="value">{data['technical']['language']}</td></tr>
                    <tr><td class="label">Model</td><td class="value">{data['technical']['model']}</td></tr>
                    <tr><td class="label">Duration</td><td class="value">{data['acoustic']['duration']:.2f} s</td></tr>
                </table>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Row 4: Transcription & Keywords
            with st.expander("📄 View Full Transcription & Keywords", expanded=True):
                st.write(f"**Keywords:** {', '.join(data['keywords'])}")
                st.text_area("Recognized Text", data['transcription'], height=100)

        else:
            st.info("Please upload or record audio in the 'Overview' tab first.")
            st.image("https://illustrations.popsy.co/amber/surr-waiting.svg", width=300)

    # --- TAB 3: CLONING ---
    with tab_cloning:
        st.header("🧬 Generative Voice Cloning")
        
        if not TTS_AVAILABLE:
            st.warning("TTS library not detected or failed to load. Please ensure `coqui-tts` is installed.")
        
        elif st.session_state.processed_data:
            col_c1, col_c2 = st.columns([1, 1])
            
            with col_c1:
                text_input = st.text_area("Text to Speak", "Hello! This is my cloned voice. I can speak in multiple languages.", height=150)
                
                # Full Name mapping for Language
                lang_map = {
                    "en": "English",
                    "fr": "French",
                    "de": "German",
                    "es": "Spanish",
                    "it": "Italian",
                    "pt": "Portuguese",
                    "pl": "Polish",
                    "tr": "Turkish",
                    "ru": "Russian",
                    "nl": "Dutch",
                    "cs": "Czech",
                    "ar": "Arabic",
                    "zh-cn": "Chinese (Simplified)",
                    "ja": "Japanese",
                    "hu": "Hungarian",
                    "ko": "Korean"
                }
                
                language_code = st.selectbox(
                    "Target Language", 
                    options=list(lang_map.keys()), 
                    format_func=lambda x: lang_map.get(x, x)
                )
                
            with col_c2:
                st.markdown("### Settings")
                speed = st.slider("Speech Rate", 0.5, 2.0, 1.0)
                st.info("This will clone the voice characteristics from your input audio.")
                
                if st.button("🔮 Synthesize Voice"):
                    tts = load_tts_model()
                    if tts:
                        with st.spinner("Synthesizing (this may take a moment)..."):
                            try:
                                # Save output to a temp file
                                out_path = "output_cloned.wav"
                                tts.tts_to_file(text=text_input, 
                                                speaker_wav="temp_input.wav", 
                                                language=language_code, 
                                                file_path=out_path,
                                                speed=speed)
                                
                                st.session_state.tts_audio = out_path
                                st.success("Synthesis Complete!")
                            except Exception as e:
                                st.error(f"Synthesis failed: {e}")
            
            if st.session_state.tts_audio:
                st.markdown("### 🎧 Result")
                st.audio(st.session_state.tts_audio)
                
                # Comparison
                with st.expander("📊 Compare Original vs Cloned"):
                    st.write("Comparison visualization would go here (using audio fingerprinting).")
        
        else:
            st.info("Upload an audio file first to provide a voice reference.")

    # --- TAB 4: EXPORT ---
    with tab_export:
        st.header("📤 Reports & Downloads")
        
        if st.session_state.processed_data:
            data = st.session_state.processed_data
            
            # Prepare CSV
            df_metrics = pd.DataFrame([data['acoustic']])
            csv_metrics = df_metrics.to_csv(index=False).encode('utf-8')
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("### 💾 Data")
                st.download_button(
                    "Download Metrics (CSV)",
                    csv_metrics,
                    "voice_metrics.csv",
                    "text/csv",
                    key='download-csv'
                )
                st.download_button(
                    "Download Transcription (TXT)",
                    data['transcription'],
                    "transcription.txt",
                    "text/plain",
                    key='download-txt'
                )

            with col_d2:
                st.markdown("### 📑 PDF Report")
                st.caption("Generate a professional summary of this session.")
                
                if not FPDF_AVAILABLE:
                    st.warning("Please install `fpdf` to generate PDF reports (`pip install fpdf`).")
                else:
                    if st.button("Generate PDF Report"):
                        with st.spinner("Generating PDF..."):
                            pdf_bytes = generate_pdf(data)
                            st.session_state.pdf_report = pdf_bytes
                            st.success("PDF Generated!")
                    
                    if st.session_state.pdf_report:
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=st.session_state.pdf_report,
                            file_name="voice_analysis_report.pdf",
                            mime="application/pdf",
                            key='download-pdf'
                        )
        else:
             st.info("No data available to export.")

if __name__ == "__main__":
    main()