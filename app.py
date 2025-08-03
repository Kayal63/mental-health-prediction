from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    stress = int(request.form['stress_level'])
    sleep = int(request.form['sleep_quality'])

    input_df = pd.DataFrame([[age, stress, sleep]], columns=['age', 'stress_level', 'sleep_quality'])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    risk = "At Risk" if pred == 1 else "Not at Risk"

    return render_template('result.html', risk=risk)

if __name__ == '__main__':
    app.run(debug=True)

