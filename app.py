import streamlit as st
import numpy as np
import pickle
import json
import time
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from streamlit_lottie import st_lottie
import sqlite3
from st_audiorec import st_audiorec
import speech_recognition as sr
from pydub import AudioSegment
import io


# --- Constants ---
MAX_LEN = 100

# --- Session Initialization ---
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'all_inputs' not in st.session_state:
    st.session_state['all_inputs'] = []

# --- Load model and tools ---
@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model("mental_health_lstm.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_and_tools()


# --- Load Lottie Animations ---
def load_lottie_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

mental_health_animation = load_lottie_file("mental_health.json")
feedback_animation = load_lottie_file("feedback.json")
loading_animation = load_lottie_file("loading.json")

# --- DB Setup ---
def init_db():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rating TEXT,
            comment TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_feedback(rating, comment):
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedback (rating, comment) VALUES (?, ?)", (rating, comment))
    conn.commit()
    conn.close()

init_db()

# --- Predict Function ---
def predict_mental_health(text):
    clean_text = ''.join(c for c in text.lower().strip() if c.isalpha() or c.isspace())
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0]

# --- Emotion Responses ---
emotion_responses = {
    "Anxiety": ("😟", "We're detecting signs of anxiety. It's okay — take deep breaths, and consider reaching out to a counselor."),
    "Stress": ("😫", "This seems stressful. Take a short break, go for a walk, or talk to someone you trust."),
    "Depression": ("😔", "We're picking up signs of depression. You're not alone. Please consider talking to a mental health professional."),
    "Anger": ("😠", "Sounds like anger. Try journaling or a calming activity."),
    "Sadness": ("😢", "Sadness is valid. Let yourself feel it, and try doing something comforting."),
    "Loneliness": ("😞", "Feeling alone can be heavy. Consider connecting with friends or a support group."),
    "Neutral": ("🙂", "You're sounding okay. Keep going and take care of yourself."),
    "Happy": ("😄", "Awesome! You sound great. Keep that energy up!"),
    "Gratitude": ("🙏", "Gratitude is powerful. Thanks for sharing your positivity!")
}

# --- Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #1c1e26;
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #fbbf24;
        font-weight: 600;
    }

    .emotion-box {
        background: linear-gradient(145deg, #232635, #1b1d27);
        box-shadow: 0 6px 16px rgba(0,0,0,0.35);
        border-radius: 20px;
        padding: 25px;
        margin-top: 30px;
        transition: transform 0.3s ease;
    }

    .emotion-box:hover {
        transform: scale(1.01);
    }

    .emoji {
        font-size: 30px;
    }

    textarea, input[type="text"] {
        background-color: #2a2b3c !important;
        color: #ffffff !important;
        font-size: 16px !important;
        border-radius: 10px !important;
        border: 1px solid #444 !important;
        padding: 10px !important;
    }

    .stButton>button {
        background-color: #fbbf24;
        color: #000;
        padding: 0.5rem 1.2rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #facc15;
        transform: scale(1.03);
    }

    .stRadio > div {
        background-color: #2a2b3c;
        border-radius: 10px;
        padding: 10px;
    }

    .stAlert {
        border-left: 6px solid #fbbf24 !important;
        background: #333444;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- New: Global Helplines ---
global_helplines = {
    "India": {
        "Depression": [
            ("📞 AASRA (24x7 Suicide Prevention)", "http://www.aasra.info"),
            ("📞 iCall – Free Psychosocial Helpline", "https://icallhelpline.org"),
            ("📞 Vandrevala Foundation Helpline", "https://www.vandrevalafoundation.com/helpline")
        ],
        "Suicidal": [
            ("📞 AASRA – 91-22-27546669", "http://www.aasra.info"),
            ("📞 iCall – Suicide Support", "https://icallhelpline.org")
        ],
        "Anxiety": [
            ("📞 Fortis Stress Helpline (91-8376804102)", "https://www.fortishealthcare.com/mental-health")
        ],
        "Stress": [
            ("📞 Mpower 1on1 (1800-120-820050)", "https://mpowerminds.com"),
            ("📞 iCall – Emotional Support", "https://icallhelpline.org")
        ]
    },
    "USA": {
        "Depression": [
            ("📞 SAMHSA’s National Helpline (1-800-662-HELP)", "https://www.samhsa.gov/find-help/national-helpline"),
            ("📞 NAMI HelpLine", "https://www.nami.org/help")
        ],
        "Suicidal": [
            ("📞 988 Suicide & Crisis Lifeline", "https://988lifeline.org"),
            ("📞 Crisis Text Line (Text HOME to 741741)", "https://www.crisistextline.org")
        ],
        "Anxiety": [
            ("📞 Anxiety and Depression Association of America", "https://adaa.org")
        ],
        "Stress": [
            ("📞 Mental Health America", "https://www.mhanational.org")
        ]
    },
    "UK": {
        "Depression": [
            ("📞 Mind Infoline (0300 123 3393)", "https://www.mind.org.uk"),
            ("📞 NHS Every Mind Matters", "https://www.nhs.uk/every-mind-matters")
        ],
        "Suicidal": [
            ("📞 Samaritans (116 123)", "https://www.samaritans.org"),
            ("📞 Shout Crisis Text Line (Text SHOUT to 85258)", "https://giveusashout.org")
        ],
        "Anxiety": [
            ("📞 Anxiety UK", "https://www.anxietyuk.org.uk")
        ],
        "Stress": [
            ("📞 Rethink Mental Illness", "https://www.rethink.org")
        ]
    }
}

# --- Main Page ---
st_lottie(mental_health_animation, height=150, key="mainpage")
st.title("🧠 Mental Health Condition Predictor")

country = st.selectbox("🌍 Select your country:", ["India", "USA", "UK"])

user_input = st.text_area("📝 Write something here:")
st.subheader("🎤 Or Record Your Voice:")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

    # Save the recording temporarily
    audio = AudioSegment.from_file(io.BytesIO(wav_audio_data), format="wav")
    audio.export("temp_audio.wav", format="wav")
    
    # Transcribe the audio
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = recognizer.record(source)
        try:
            text_from_audio = recognizer.recognize_google(audio_data)
            st.success(f"📝 Transcribed Text: {text_from_audio}")

            # Optionally: Autofill the text input
            if 'all_inputs' in st.session_state:
                st.session_state.all_inputs.append(text_from_audio)
            user_input = text_from_audio  # Overwrite the earlier text input
            st.session_state['user_input_from_voice'] = text_from_audio
        except sr.UnknownValueError:
            st.error("❌ Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"❌ Error from Google Speech Recognition service; {e}")


if st.button("🧠 Analyze My Mental State"):
    if user_input.strip():
        st.session_state.analyzed = True
        st.session_state.all_inputs.append(user_input)
        st.session_state['result'] = predict_mental_health(user_input)
        st.session_state['country'] = country
        st.rerun()

# --- After Prediction Flow ---
if st.session_state.analyzed:
    result = st.session_state.get("result")
    emoji, message = emotion_responses.get(result, ("🧐", "Unable to classify this feeling."))
    selected_country = st.session_state.get("country", "India")

    st.markdown(f"""
        <div class="emotion-box">
            <h3 class="emoji">{emoji} Detected Condition: {result}</h3>
            <p>{message}</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🆘 Get Help Now")
    st.markdown(f"These helplines are available in **{selected_country}** based on what you're feeling:")

    if result in global_helplines.get(selected_country, {}):
        for title, link in global_helplines[selected_country][result]:
            st.markdown(f"- [{title}]({link})", unsafe_allow_html=True)
    else:
        st.info("No specific helpline available for this condition, but general support is always available.")
        # Optionally show default support numbers
        for emotion_links in global_helplines[selected_country].values():
            for title, link in emotion_links[:1]:  # Just show one from each category
                st.markdown(f"- {title}: [Visit]({link})", unsafe_allow_html=True)

    # Feedback Section
    st.subheader("💬 Was this prediction helpful?")
    feedback_rating = st.radio("Rate the prediction accuracy:", ["Very Accurate", "Somewhat Accurate", "Not Accurate"])
    feedback_comment = st.text_input("Leave an optional comment:")

    if st.button("Submit Feedback"):
        store_feedback(feedback_rating, feedback_comment)
        st.success("✅ Thank you for your feedback!")
        time.sleep(2)
        st.session_state.analyzed = False
        st.session_state['result'] = None
        st.rerun()
