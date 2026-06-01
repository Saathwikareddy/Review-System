import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Configuration
# -------------------------------------------------

VOCAB_SIZE = 10000
MAX_LEN = 200

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

# -------------------------------------------------
# Load Word Index
# -------------------------------------------------

@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    return word_index

word_index = load_word_index()

# -------------------------------------------------
# Text to Sequence
# -------------------------------------------------

def review_to_sequence(review):

    review = review.lower().split()

    sequence = []

    for word in review:
        if word in word_index:
            sequence.append(word_index[word] + 3)
        else:
            sequence.append(2)

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded

# -------------------------------------------------
# Load Models
# -------------------------------------------------

@st.cache_resource
def load_models():

    rnn_model = tf.keras.models.load_model(
        "simple_rnn_model.h5"
    )

    lstm_model = tf.keras.models.load_model(
        "lstm_model.h5"
    )

    gru_model = tf.keras.models.load_model(
        "gru_model.h5"
    )

    return rnn_model, lstm_model, gru_model

model_rnn, model_lstm, model_gru = load_models()

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------

def predict_sentiment(model, review):

    sequence = review_to_sequence(review)

    pred = model.predict(
        sequence,
        verbose=0
    )[0][0]

    positive_prob = float(pred)
    negative_prob = float(1 - pred)

    sentiment = (
        "Positive"
        if positive_prob >= 0.5
        else "Negative"
    )

    confidence = max(
        positive_prob,
        negative_prob
    ) * 100

    return (
        sentiment,
        confidence,
        positive_prob,
        negative_prob
    )

# -------------------------------------------------
# Header
# -------------------------------------------------

st.title(
    "🎬 Movie Review Sentiment Analysis System"
)

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# -------------------------------------------------
# Model Selection
# -------------------------------------------------

selected_model = st.radio(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

# -------------------------------------------------
# Input Area
# -------------------------------------------------

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# -------------------------------------------------
# Predict Button
# -------------------------------------------------

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    model_map = {
        "SimpleRNN": model_rnn,
        "LSTM": model_lstm,
        "GRU": model_gru
    }

    model = model_map[selected_model]

    sentiment, confidence, pos_prob, neg_prob = (
        predict_sentiment(
            model,
            review
        )
    )

    st.markdown("---")

    st.success(
        f"Sentiment: {sentiment}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # ---------------------------------------------
    # Probability Chart
    # ---------------------------------------------

    st.subheader("Probability Distribution")

    fig, ax = plt.subplots(figsize=(5,3))

    labels = ["Positive", "Negative"]
    values = [pos_prob, neg_prob]

    ax.bar(labels, values)

    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")

    st.pyplot(fig)

    # ---------------------------------------------
    # Compare All Models
    # ---------------------------------------------

    st.markdown("---")

    st.subheader(
        "Comparison Across All Models"
    )

    results = []

    for name, model in [
        ("SimpleRNN", model_rnn),
        ("LSTM", model_lstm),
        ("GRU", model_gru)
    ]:

        sentiment, confidence, pos_prob, neg_prob = (
            predict_sentiment(
                model,
                review
            )
        )

        results.append({
            "Model": name,
            "Sentiment": sentiment,
            "Confidence (%)": round(
                confidence,
                2
            ),
            "Positive Probability": round(
                pos_prob,
                4
            ),
            "Negative Probability": round(
                neg_prob,
                4
            )
        })

    st.dataframe(
        results,
        use_container_width=True
    )

    # ---------------------------------------------
    # Confidence Chart
    # ---------------------------------------------

    st.subheader(
        "Confidence Comparison"
    )

    fig2, ax2 = plt.subplots(
        figsize=(6,4)
    )

    model_names = [
        r["Model"]
        for r in results
    ]

    confidences = [
        r["Confidence (%)"]
        for r in results
    ]

    ax2.bar(
        model_names,
        confidences
    )

    ax2.set_ylabel(
        "Confidence (%)"
    )

    ax2.set_ylim(
        [0, 100]
    )

    st.pyplot(fig2)

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("---")
st.caption(
    "Sentiment Analysis using SimpleRNN, LSTM and GRU"
)
