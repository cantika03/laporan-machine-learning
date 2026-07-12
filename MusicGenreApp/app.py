import streamlit as st
import pandas as pd
import joblib

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Music Genre Prediction",
    page_icon="🎵",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df = pd.read_csv("tcc_ceds_music.csv")

# ==========================
# Header
# ==========================
st.title("🎵 Music Genre Prediction")
st.caption("Prediksi genre lagu berdasarkan lirik menggunakan Machine Learning (Naive Bayes)")

st.divider()

# ==========================
# Statistik
# ==========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎵 Total Lagu", len(df))

with col2:
    st.metric("🎼 Total Genre", df["genre"].nunique())

with col3:
    st.metric("🤖 Model", "Naive Bayes")

st.divider()

# ==========================
# Input Lirik
# ==========================
st.subheader("📝 Masukkan Lirik Lagu")

lyrics = st.text_area(
    "Lirik",
    height=220,
    placeholder="Masukkan atau tempel lirik lagu di sini..."
)

if st.button("🎵 Prediksi Genre", use_container_width=True):

    if lyrics.strip() == "":
        st.warning("Silakan masukkan lirik lagu terlebih dahulu.")
    else:

        vector = vectorizer.transform([lyrics])
        prediction = model.predict(vector)[0]

        st.success("Prediksi berhasil!")

        st.metric(
            label="Genre Hasil Prediksi",
            value=prediction
        )

        st.balloons()

st.divider()

# ==========================
# Distribusi Genre
# ==========================
st.subheader("📊 Distribusi Genre")

genre = df["genre"].value_counts()

st.bar_chart(genre)

st.divider()

# ==========================
# Preview Dataset
# ==========================
with st.expander("📄 Lihat Dataset"):

    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ==========================
# Tentang
# ==========================
st.subheader("👩‍💻 Tentang Aplikasi")

st.write("""
Aplikasi ini dibuat untuk memprediksi **genre lagu berdasarkan lirik**
menggunakan algoritma **Naive Bayes** dengan representasi fitur **TF-IDF**.
""")

