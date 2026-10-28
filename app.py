from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model_path = MODEL_DIR / "mlp_pipeline.pkl"
features_path = MODEL_DIR / "features.json"
metrics_path = MODEL_DIR / "metrics.json"

for path in (model_path, features_path, metrics_path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero las celdas de entrenamiento del notebook."
        )

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(features_path, "r", encoding="utf-8") as f:
    features = json.load(f)
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predicción de aprobación estudiantil - MLP</title>
<style>
body { font-family: Arial, sans-serif; max-width: 760px; margin: 30px auto; padding: 16px; background: #f4f6f8; color: #222; }
.card { background: white; padding: 24px; border-radius: 10px; box-shadow: 0 2px 10px #0001; }
h1 { text-align: center; font-size: 1.6rem; }
.form-group { margin: 14px 0; }
label { display: block; font-weight: bold; margin-bottom: 6px; }
input, select { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #bbb; border-radius: 5px; }
button { width: 100%; margin-top: 10px; padding: 12px; border: 0; border-radius: 5px; color: white; background: #1769aa; font-size: 1rem; cursor: pointer; }
.result { margin-top: 18px; padding: 16px; border-radius: 6px; text-align: center; }
.aprobado { background: #d9f2df; color: #155724; }
.reprobado { background: #f8dadd; color: #721c24; }
.error { background: #fff0cc; color: #694d00; }
.metrics { margin-top: 20px; padding: 14px; border-radius: 6px; background: #eaf2fb; }
</style>
</head>
<body>
<div class="card">
<h1>🎓 Predicción de aprobación estudiantil (MLP)</h1>
<form id="predForm">
<div class="form-group"><label for="Tareas">Tareas (0-100)</label><input id="Tareas" type="number" name="Tareas" min="0" max="100" step="any" required></div>
<div class="form-group"><label for="Examenes">Exámenes (0-100)</label><input id="Examenes" type="number" name="Examenes" min="0" max="100" step="any" required></div>
<div class="form-group"><label for="Asistencia">Asistencia (%)</label><input id="Asistencia" type="number" name="Asistencia" min="0" max="100" step="any" required></div>
<div class="form-group"><label for="Estudio_Horas">Horas de estudio por semana</label><input id="Estudio_Horas" type="number" name="Estudio_Horas" min="0" max="168" step="any" required></div>
<div class="form-group"><label for="Sueno_Horas">Horas de sueño diarias</label><input id="Sueno_Horas" type="number" name="Sueno_Horas" min="0" max="24" step="any" required></div>
<div class="form-group"><label for="Extracurriculares">Participación extracurricular</label><select id="Extracurriculares" name="Extracurriculares" required><option value="1">Sí (1)</option><option value="0">No (0)</option></select></div>
<button type="submit">Predecir</button>
</form>
<div id="result" aria-live="polite"></div>
<div class="metrics"><strong>Métricas en el conjunto de prueba:</strong><br>
Accuracy: {{ metrics.accuracy|round(4) }} · Precision: {{ metrics.precision|round(4) }} · Recall: {{ metrics.recall|round(4) }} · F1-score: {{ metrics.f1|round(4) }}
</div>
</div>
<script>
document.getElementById('predForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData.entries());
  for (const key of Object.keys(data)) data[key] = Number(data[key]);
  const box = document.getElementById('result');
  box.textContent = 'Calculando predicción...';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'No fue posible calcular la predicción.');
    const className = result.prediccion === 1 ? 'aprobado' : 'reprobado';
    box.innerHTML = `<div class="result ${className}"><strong>Predicción:</strong> ${result.clase}<br><strong>Probabilidad estimada de aprobar:</strong> ${(result.prob_aprobar * 100).toFixed(2)}%</div>`;
  } catch (err) {
    box.innerHTML = `<div class="result error">${err.message}</div>`;
  }
});
</script>
</body>
</html>
"""

def parse_features(data):
    if not isinstance(data, dict):
        raise ValueError("El cuerpo de la solicitud debe ser un objeto JSON.")
    missing = [f for f in features if f not in data]
    if missing:
        raise ValueError(f"Faltan campos requeridos: {missing}")
    try:
        values = {f: float(data[f]) for f in features}
    except (TypeError, ValueError):
        raise ValueError("Todos los campos deben contener valores numéricos.")
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Los valores deben ser números finitos.")
    for f in ("Tareas", "Examenes", "Asistencia"):
        if not 0 <= values[f] <= 100:
            raise ValueError(f"{f} debe estar entre 0 y 100.")
    if not 0 <= values["Estudio_Horas"] <= 168:
        raise ValueError("Estudio_Horas debe estar entre 0 y 168.")
    if not 0 <= values["Sueno_Horas"] <= 24:
        raise ValueError("Sueno_Horas debe estar entre 0 y 24.")
    if values["Extracurriculares"] not in (0, 1):
        raise ValueError("Extracurriculares debe ser 0 o 1.")
    return pd.DataFrame([[values[f] for f in features]], columns=features)

@app.get("/")
def home():
    return render_template_string(HTML, metrics=metrics)

@app.post("/predict")
@app.post("/api/predict")
def predict():
    try:
        X = parse_features(request.get_json(silent=True))
        pred = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        class_to_index = {int(c): i for i, c in enumerate(model.classes_)}
        return jsonify({
            "prediccion": pred,
            "clase": "Aprobado" if pred == 1 else "Reprobado",
            "prob_aprobar": float(probabilities[class_to_index[1]]),
            "prob_reprobar": float(probabilities[class_to_index[0]])
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "MLPClassifier", "loaded": True})

if __name__ == "__main__":
    
    app.run(host="127.0.0.1", port=5000, debug=False)
