from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo y escalador previamente guardados
modelo = joblib.load('modelo_red_neuronal.pkl')
escalador = joblib.load('escalador.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # asegúrate que tu HTML se llame así

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario HTML (4 variables)
        valores = [
            float(request.form['Rainfall_mm']),
            float(request.form['Fertilizer_Used']),
            float(request.form['Irrigation_Used']),
            float(request.form['Temperature_Celsius']),
        ]

        # Escalar los datos antes de predecir
        datos_escalados = escalador.transform([valores])

        # Realizar predicción
        prediccion = modelo.predict(datos_escalados)

        # Devolver el resultado como JSON
        return jsonify({'prediccion': round(prediccion[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
