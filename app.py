import streamlit as st
import pickle
import time
import nltk

# Ensure necessary NLTK resources are available before execution
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def transform_text(text):
    # Initialize the stemmer
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))  # Convert to set for faster lookups
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Removing special characters, stop words, and punctuation + Stemming
    processed_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]

    # Return the cleaned and transformed text
    return " ".join(processed_tokens)

# Define custom menu items
custom_menu_items = {
    'Report a bug': "mailto:babuldashbabul@gmail.com",
    'About': "This is a spam classifier web app"
}

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="📧 Email/SMS Spam Classifier",
    page_icon="📩",
    menu_items=custom_menu_items
)

# Load vectorizer & model
if "vectorizer" not in st.session_state:
    st.session_state['vectorizer'] = pickle.load(open('vectorizer.pkl', 'rb'))
if "model" not in st.session_state:
    st.session_state['model'] = pickle.load(open('model.pkl', 'rb'))

tfidf = st.session_state['vectorizer']
model = st.session_state['model']

# Custom Styles
st.markdown("""
    <style>
        .stApp {
            background-color: #1f1f1f;
        }
        .main-title {
            font-size: 38px;
            font-weight: bold;
            color: #007BFF;
            text-align: center;
        }
        .sub-title {
            font-size: 18px;
            color: #555;
            text-align: center;
        }
        .result-box {
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        .spam {
            background-color: #ffcccc;
            color: #d9534f;
        }
        .not-spam {
            background-color: #ccffcc;
            color: #28a745;
        }
        .centered {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">📧 Email/SMS Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A simple AI-powered tool to detect spam messages</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📩 Direct Input", "📂 Upload File"])

# Direct input method
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_sms = st.text_area("Enter the message", placeholder="Type here...", height=150)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3500/3500833.png", width=150)  # Example Image

    st.divider()

    if st.button('🔍 Predict'):
        if not input_sms.strip():
            st.warning("⚠️ Please enter a message to predict.")
            st.stop()

        # Processing animation
        with st.status("⏳ Analyzing...", expanded=True) as status:
            st.write("📄 Preprocessing text...")
            time.sleep(1.5)
            st.write("🔢 Converting to vector format...")
            time.sleep(1.5)
            st.write("🤖 Running classification model...")
            time.sleep(1.5)

            # Predict result
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            status.update(label="✅ Analysis Complete!", state="complete", expanded=True)

            if result == 1:
                st.markdown('<div class="result-box spam">🚨 This message is SPAM!</div>', unsafe_allow_html=True)
                st.toast("⚠️ Be cautious, this might be spam!", icon='🚨')
            else:
                st.markdown('<div class="result-box not-spam">🎉 This message is NOT Spam!</div>', unsafe_allow_html=True)
                st.toast("✅ Safe message detected!", icon='🎉')

# File upload method
with tab2:
    uploaded_file = st.file_uploader("📂 Upload a text file", type="txt")

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.toast("Upload a valid text file", icon='🚨')
            st.stop()

        st.success("📄 File uploaded successfully!")

        transformed_sms = transform_text(file_content)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.divider()

        if result == 1:
            st.markdown('<div class="result-box spam">🚨 This message is SPAM!</div>', unsafe_allow_html=True)
            st.toast("⚠️ Be cautious, this might be spam!", icon='🚨')
        else:
            st.markdown('<div class="result-box not-spam">🎉 This message is NOT Spam!</div>', unsafe_allow_html=True)
            st.toast("✅ Safe message detected!", icon='🎉')

st.divider()
st.markdown("💡 **Tip:** If a message is spam, never click on suspicious links or share personal information!")
