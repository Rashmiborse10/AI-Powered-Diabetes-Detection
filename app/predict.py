import time
import streamlit as st
from loader import model, accuracy_result
from data.config import thresholds
from function.function import make_donut
from data.base import mrk
from app.health_guidance import get_risk_guidance, guidance_to_markdown


def app(input_data):
    prediction = model.predict_proba(input_data)[:, 1]
    probability = float(prediction[0])
    is_diabetes = probability >= thresholds

    # Store prediction context for the AI chatbot
    st.session_state.prediction_context = {
        'probability': round(probability * 100, 2),
        'is_diabetes': is_diabetes,
        'inputs': {
            'Pregnancies': float(input_data.iloc[0]['Pregnancies']),
            'Glucose': float(input_data.iloc[0]['Glucose']),
            'Insulin': float(input_data.iloc[0]['Insulin']),
            'BMI': float(input_data.iloc[0]['BMI']),
            'Age': float(input_data.iloc[0]['Age']),
        }
    }

    cols = st.columns(2)
    guidance = get_risk_guidance(round(probability * 100, 2), is_diabetes, st.session_state.prediction_context['inputs'])

    def stream_data():
        risk_label = 'Diabetes' if is_diabetes else 'No Diabetes'
        text = f"Model Accuracy: {accuracy_result}%\n\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)
        text = f"\nPrediction: {risk_label}\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)
        text = f"\nProbability: {round(probability * 100, 2)}%\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)
        
        return 80

    cols[0].write_stream(stream_data)


    is_diabetes_text = f'<strong>Warning:</strong> Diabetes!' if is_diabetes else 'No Diabetes'
    color = 'red' if is_diabetes else 'blue'

    cols[1].markdown(mrk.format(color, is_diabetes_text), unsafe_allow_html=True)
    cols[1].write('\n\n\n\n\n')
    donut_chart_population = make_donut(round(probability * 100, 2), 
                                        'Diabetes Risk',
                                        input_color=color)

    cols[1].altair_chart(donut_chart_population)

    st.markdown("### What this result means")
    st.markdown(guidance_to_markdown(guidance))
