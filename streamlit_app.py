
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'modelo_desercion.pkl'
loaded_model = joblib.load(model_filename)

st.set_page_config(page_title="Student Dropout Prediction", layout="centered")
st.title("🎓 Student Dropout Prediction")
st.markdown("--- Say Goodbye to Dropouts! --- ")

st.markdown(
    "This application predicts the likelihood of a student dropping out based on various academic and socioeconomic factors. "
    "Adjust the sliders and select boxes to input student information and see the prediction."
)

st.sidebar.header("Student Information Input")

# Input widgets for each feature
edad = st.sidebar.slider('Age (edad)', min_value=18, max_value=29, value=22)
promedio = st.sidebar.slider('Average Grade (promedio)', min_value=2.0, max_value=5.0, value=3.5, step=0.01)
asistencia = st.sidebar.slider('Attendance Percentage (asistencia)', min_value=50, max_value=100, value=85)
horas_estudio = st.sidebar.slider('Study Hours per Week (horas_estudio)', min_value=0, max_value=39, value=20)
uso_plataforma = st.sidebar.slider('Platform Usage Hours per Week (uso_plataforma)', min_value=0, max_value=14, value=7)
materias_perdidas = st.sidebar.slider('Failed Courses (materias_perdidas)', min_value=0, max_value=5, value=1)
nivel_socioeconomico = st.sidebar.selectbox('Socioeconomic Level (nivel_socioeconomico)', options=[1, 2, 3, 4, 5], index=2)
trabaja_map = {0: 'No', 1: 'Yes'}
trabaja_input = st.sidebar.selectbox('Works (trabaja)', options=list(trabaja_map.keys()), format_func=lambda x: trabaja_map[x])
acceso_internet_map = {0: 'No', 1: 'Yes'}
acceso_internet_input = st.sidebar.selectbox('Internet Access (acceso_internet)', options=list(acceso_internet_map.keys()), format_func=lambda x: acceso_internet_map[x])

# Create a DataFrame from inputs
input_data = pd.DataFrame([{
    'edad': edad,
    'promedio': promedio,
    'asistencia': asistencia,
    'horas_estudio': horas_estudio,
    'uso_plataforma': uso_plataforma,
    'materias_perdidas': materias_perdidas,
    'nivel_socioeconomico': nivel_socioeconomico,
    'trabaja': trabaja_input,
    'acceso_internet': acceso_internet_input
}])

st.subheader("Input Data for Prediction")
st.write(input_data)

# Make prediction
if st.button('Predict Dropout'):
    prediction = loaded_model.predict(input_data)
    prediction_proba = loaded_model.predict_proba(input_data)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"The student is predicted to DROP OUT. (Probability: {prediction_proba[0][1]:.2f})")
        st.markdown("⚠️ **Recommendation:** Consider providing immediate academic and personal support. Identify potential causes like low grades, poor attendance, or socioeconomic challenges and offer targeted interventions.")
    else:
        st.success(f"The student is predicted NOT to drop out. (Probability: {prediction_proba[0][0]:.2f})")
        st.markdown("✅ **Recommendation:** Continue monitoring student progress and engagement to ensure continued success.")
