import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset for training
training_data = pd.DataFrame({
    'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
    'Attendance': [50,60,65,70,75,80,85,90,92,95],
    'Assignments_Submitted': [0,1,1,2,3,3,4,4,5,5],
    'Participation': [1,2,2,3,3,4,4,5,5,5],
    'Score': [35,40,45,50,55,60,65,75,80,90]
})

# Train the model
X = training_data[['Hours_Studied', 'Attendance', 'Assignments_Submitted', 'Participation']]
y = training_data['Score']
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Student Score Predictor", page_icon="📊")
st.title("🎓 Student Score Predictor")
st.markdown("Predict a student's exam score out of 100 using ML and user input.")

st.sidebar.header("📥 Enter Student Details")

hours = st.sidebar.number_input("Hours Studied per Week", min_value=0, max_value=20, value=5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
assignments = st.sidebar.slider("Assignments Submitted", 0, 10, 3)
participation = st.sidebar.slider("Participation (1-5)", 1, 5, 3)

if st.sidebar.button("Predict Score"):
    input_df = pd.DataFrame([[hours, attendance, assignments, participation]],
                            columns=['Hours_Studied', 'Attendance', 'Assignments_Submitted', 'Participation'])
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Exam Score: **{prediction:.2f} / 100**")

st.markdown("---")
st.markdown("Built using **Python + Streamlit + Scikit-learn**.")
