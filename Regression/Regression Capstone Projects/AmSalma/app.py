import joblib
import numpy as np
import gradio as gr
import os

MODEL_PATH = os.path.join('models', 'best_model.pkl')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first to create it.")

model = joblib.load(MODEL_PATH)

def predict_medical_cost(age, sex, bmi, children, smoker,
                         region_northwest, region_southeast, region_southwest):
    # inputs come as strings from dropdowns for sex/smoker in this app, ensure ints
    sex = int(sex)
    smoker = int(smoker)
    input_data = np.array([[age, sex, bmi, children, smoker,
                            region_northwest, region_southeast, region_southwest]])
    prediction = model.predict(input_data)[0]
    return f"💰 التكلفة المتوقعة للتأمين الطبي: {prediction:.2f} دولار"

app = gr.Interface(
    fn=predict_medical_cost,
    inputs=[
        gr.Number(label="Age", value=35),
        gr.Dropdown(choices=["0","1"], label="Sex (0: Female, 1: Male)", value="1"),
        gr.Number(label="BMI", value=30.0),
        gr.Number(label="Children", value=0),
        gr.Dropdown(choices=["0","1"], label="Smoker (0: No, 1: Yes)", value="0"),
        gr.Number(label="Region_Northwest", value=0),
        gr.Number(label="Region_Southeast", value=0),
        gr.Number(label="Region_Southwest", value=0)
    ],
    outputs=gr.Textbox(label="Predicted Cost"),
    title="🏥 Medical Cost Prediction App",
    description="نموذج لتوقع تكلفة التأمين الطبي.",
)

if __name__ == '__main__':
    app.launch()
