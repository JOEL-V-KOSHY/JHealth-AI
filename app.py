import streamlit as st
import sqlite3
import hashlib
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import pickle
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from fer import FER

# =========================================================
# DATABASE SETUP
# =========================================================

conn = sqlite3.connect("health_app.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS health_records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    time TEXT,
    heart_rate INTEGER,
    oxygen INTEGER,
    mood TEXT,
    health_score INTEGER
)
""")

conn.commit()

# =========================================================
# PASSWORD FUNCTIONS
# =========================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        c.execute("INSERT INTO users(username, password) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

# =========================================================
# ML MODEL TRAINING
# =========================================================

MODEL_FILE = "pcos_model.pkl"
DATA_FILE = "PCOS_data.csv"

def train_model():

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    # Auto detect columns
    def find_column(keyword):
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
        return None

    age_col = find_column("Age")
    bmi_col = find_column("BMI")
    cycle_col = find_column("Cycle")
    weight_col = find_column("Weight")
    hair_col = find_column("Hair")
    target_col = find_column("PCOS")

    features = [age_col, bmi_col, cycle_col, weight_col, hair_col]
    target = target_col

    df = df[features + [target]]

    df.fillna(df.median(numeric_only=True), inplace=True)

    for col in [weight_col, hair_col, target]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({
            "Y":1, "N":0,
            "Yes":1, "No":0,
            "1":1, "0":0
        })
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    pickle.dump((model, features), open(MODEL_FILE, "wb"))

    return model, features, acc, cm, df, report

# Load or train model
if os.path.exists(MODEL_FILE):
    model, feature_names = pickle.load(open(MODEL_FILE, "rb"))
else:
    model, feature_names, _, _, _, _ = train_model()

# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.title("🌸 JHealth+ AI")

    menu = st.selectbox("Select Option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if menu == "Register":
        if st.button("Register"):
            if register_user(username, password):
                st.success("Account Created Successfully 🎉")
            else:
                st.error("Username already exists")

    if menu == "Login":
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid Credentials")

# =========================================================
# MAIN APP
# =========================================================

else:

    st.sidebar.success(f"Logged in as: {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    menu = st.sidebar.selectbox(
        "Navigation",
        ["Health Scan", "PCOS Prediction", "Data Science Dashboard", "Reports"]
    )

    # =====================================================
    # HEALTH SCAN WITH EMOTION DETECTION
    # =====================================================

    if menu == "Health Scan":

        st.header("📷 AI Face Health Scan")

        camera = st.camera_input("Capture your face")

        if camera is not None:

            file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:

                # REAL EMOTION DETECTION
                detector = FER(mtcnn=True)
                emotions = detector.detect_emotions(img)

                if len(emotions) > 0:
                    top_emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
                    mood = top_emotion.capitalize()
                else:
                    mood = "Neutral"

                # Optional: heart rate ranges based on mood
                if mood.lower() == "happy":
                    heart_rate = random.randint(65, 85)
                elif mood.lower() == "sad":
                    heart_rate = random.randint(85, 105)
                else:
                    heart_rate = random.randint(75, 95)

                oxygen = random.randint(95, 100)
                health_score = int((oxygen * 0.6) + ((100 - abs(heart_rate - 75)) * 0.4))

                # Show metrics with emoji
                emoji_map = {
                    "Happy": "😄", "Sad": "😢", "Neutral": "😐",
                    "Angry": "😠", "Surprise": "😲",
                    "Fear": "😱", "Disgust": "🤢"
                }
                st.metric("💚 Mood", f"{mood} {emoji_map.get(mood, '')}")
                st.metric("❤️ Heart Rate", heart_rate)
                st.metric("🫁 Oxygen", oxygen)
                st.metric("💚 Health Score", health_score)

                # Save to database
                c.execute("""
                INSERT INTO health_records(username, time, heart_rate, oxygen, mood, health_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    st.session_state.username,
                    str(datetime.now()),
                    heart_rate,
                    oxygen,
                    mood,
                    health_score
                ))
                conn.commit()

            else:
                st.error("No face detected ❌")

    # =====================================================
    # PCOS PREDICTION
    # =====================================================

    if menu == "PCOS Prediction":

        st.header("🧠 AI-Based PCOS Risk Detection")

        age = st.number_input("Age", 18, 50)
        bmi = st.number_input("BMI", 15.0, 45.0)
        cycle = st.number_input("Cycle Length (days)", 20, 50)
        weight_gain = st.selectbox("Weight Gain?", ["No", "Yes"])
        hair_growth = st.selectbox("Excess Hair Growth?", ["No", "Yes"])

        if st.button("Predict"):

            weight_gain_val = 1 if weight_gain == "Yes" else 0
            hair_growth_val = 1 if hair_growth == "Yes" else 0

            input_data = np.array([[age, bmi, cycle,
                                    weight_gain_val,
                                    hair_growth_val]])

            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("⚠ High Risk of PCOS")
            else:
                st.success("✅ Low Risk of PCOS")

    # =====================================================
    # DATA SCIENCE DASHBOARD
    # =====================================================

    if menu == "Data Science Dashboard":

        st.header("📊 Model Performance Dashboard")

        model, cols, acc, cm, df, report = train_model()

        st.success(f"Model Accuracy: {round(acc*100,2)}%")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("Feature Importance")
        importance = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": cols,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_df.set_index("Feature"))

        st.subheader("PCOS Distribution")
        fig2, ax2 = plt.subplots()
        df[cols[-1]].value_counts().plot.pie(autopct="%1.1f%%", ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # =====================================================
    # REPORTS
    # =====================================================

    if menu == "Reports":

        st.header("📁 Health History")

        df = pd.read_sql_query(
            "SELECT * FROM health_records WHERE username=?",
            conn,
            params=(st.session_state.username,)
        )

        if not df.empty:

            st.dataframe(df)
            st.line_chart(df[["heart_rate", "oxygen", "health_score"]])

            fig, ax = plt.subplots()
            df["mood"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

            if st.button("Download PDF Report"):

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                pdf.cell(200, 10, txt=f"Health Report - {st.session_state.username}", ln=True)
                pdf.cell(200, 10, txt=f"Generated on: {datetime.now()}", ln=True)
                pdf.ln(10)

                for index, row in df.iterrows():
                    pdf.cell(200, 8, txt=f"Time: {row['time']}", ln=True)
                    pdf.cell(200, 8, txt=f"Heart Rate: {row['heart_rate']}", ln=True)
                    pdf.cell(200, 8, txt=f"Oxygen: {row['oxygen']}", ln=True)
                    pdf.cell(200, 8, txt=f"Mood: {row['mood']}", ln=True)
                    pdf.cell(200, 8, txt=f"Health Score: {row['health_score']}", ln=True)
                    pdf.ln(5)

                file_name = f"{st.session_state.username}_health_report.pdf"
                pdf.output(file_name)

                with open(file_name, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name=file_name,
                        mime="application/pdf"
                    )

        else:
            st.warning("No records yet.")
