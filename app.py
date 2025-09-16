
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import sqlite3

# ---- Load model, scaler, encoders, feature columns ----
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder (2).pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ---- SQLite: Save patient data ----
def save_patient_to_db(input_data, predicted_time):
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_records (
            Age INTEGER,
            Gender TEXT,
            EducationLevel TEXT,
            BMI REAL,
            SleepQuality INTEGER,
            Depression INTEGER,
            PhysicalActivity INTEGER,
            MemoryComplaints INTEGER,
            Forgetfulness INTEGER,
            Confusion INTEGER,
            Disorientation INTEGER,
            PersonalityChanges INTEGER,
            DifficultyCompletingTasks INTEGER,
            PredictedLeaveTime TEXT
        )
    ''')
    
    input_data['PredictedLeaveTime'] = predicted_time
    c.execute('''
        INSERT INTO patient_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', tuple(input_data.values()))
    
    conn.commit()
    conn.close()

# ---- Email sending function ----
def send_email(receiver_email, predicted_time):
    sender_email = "yelalanikitha@gmail.com"        # replace with your email
    sender_password = "kukj yzan obdh ssuw"        # replace with App Password

    subject = "Alzheimer Patient Alert"
    body = f"⏰ The system predicts the patient may leave home at: {predicted_time}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False

# ---- Prediction function ----
def predict_leave_time(input_data, model, encoders, scaler, X_columns, wake_time_str):
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    for col, le in encoders.items():
        if col in input_df.columns:
            val = input_df[col].values[0]
            input_df[col] = le.transform([val]) if val in le.classes_ else 0

    # Fill missing columns
    for col in X_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X_columns]
    input_scaled = scaler.transform(input_df)

    predicted_minutes = model.predict(input_scaled)[0]
    predicted_time = datetime.strptime(wake_time_str, "%H:%M") + timedelta(minutes=int(round(predicted_minutes)))
    return predicted_time.strftime("%I:%M %p")

# ---- Streamlit UI ----
st.title("Alzheimer's Patient Leave Time Predictor & Alert System")

st.subheader("Enter Patient Details")
with st.form("patient_form"):
    age = st.number_input("Age", min_value=50, max_value=120, value=75)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    sleep = st.slider("Sleep Quality (1-10)", 1, 10, 6)
    depression = st.slider("Depression (0-10)", 0, 10, 1)
    physical = st.slider("Physical Activity (1-5)", 1, 5, 2)
    memory = st.slider("Memory Complaints (1-10)", 1, 10, 5)
    forgetfulness = st.slider("Forgetfulness (1-10)", 1, 10, 5)
    confusion = st.slider("Confusion (1-10)", 1, 10, 3)
    disorientation = st.slider("Disorientation (1-10)", 1, 10, 3)
    personality = st.slider("Personality Changes (1-10)", 1, 10, 2)
    difficulty = st.slider("Difficulty Completing Tasks (1-10)", 1, 10, 3)
    wake_time = st.time_input("Wake Up Time", value=datetime.strptime("07:00", "%H:%M").time())
    family_email = st.text_input("Family Email")

    submitted = st.form_submit_button("Predict & Send Email")

if submitted:
    input_data = {
        "Age": age,
        "Gender": gender,
        "EducationLevel": education,
        "BMI": bmi,
        "SleepQuality": sleep,
        "Depression": depression,
        "PhysicalActivity": physical,
        "MemoryComplaints": memory,
        "Forgetfulness": forgetfulness,
        "Confusion": confusion,
        "Disorientation": disorientation,
        "PersonalityChanges": personality,
        "DifficultyCompletingTasks": difficulty,
    }

    wake_time_str = wake_time.strftime("%H:%M")
    predicted_time_str = predict_leave_time(input_data, model, label_encoders, scaler, feature_columns, wake_time_str)
    
    st.success(f"⏰ Predicted Leave Time: {predicted_time_str}")

    # Save to SQLite
    save_patient_to_db(input_data, predicted_time_str)

    # Send email
    if family_email:
        if send_email(family_email, predicted_time_str):
            st.info(f"📧 Email sent to {family_email} successfully!")



