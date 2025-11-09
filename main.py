"""
Servicio de Predicción de Enfermedades usando Random Forest
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Disease Predictor API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para modelo y síntomas
model = None
symptoms_list = []

# Modelo de datos para la petición
class PredictionRequest(BaseModel):
    sintomas: List[str]
    consentimiento: bool = True

class PredictionResponse(BaseModel):
    predicciones: List[Dict[str, Any]]
    recomendaciones: List[str]
    urgencia: str
    disclaimer: str

def load_model_and_symptoms():
    """Carga el modelo y la lista de síntomas"""
    global model, symptoms_list

    try:
        # Intentar cargar el modelo
        model_path = "modelo_enfermedades.pkl"
        symptoms_path = "symptoms_list.pkl"

        if os.path.exists(model_path) and os.path.exists(symptoms_path):
            logger.info("Cargando modelo y síntomas desde archivos...")
            model = joblib.load(model_path)
            symptoms_list = joblib.load(symptoms_path)
            logger.info(f"Modelo cargado. Síntomas disponibles: {len(symptoms_list)}")
        else:
            logger.warning("Modelo no encontrado. Usando modelo de prueba.")
            # Crear modelo de prueba con síntomas básicos
            from sklearn.ensemble import RandomForestClassifier
            symptoms_list = [
                'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
                'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
                'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
                'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
                'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
                'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
                'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
                'dehydration', 'indigestion', 'headache', 'yellowish_skin',
                'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
                'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
                'mild_fever', 'yellow_urine'
            ]
            # Crear un modelo simple de prueba
            model = RandomForestClassifier(n_estimators=10, random_state=42)

    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("Iniciando servicio de predicción de enfermedades...")
    load_model_and_symptoms()
    logger.info("Servicio iniciado correctamente")

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Disease Predictor API",
        "version": "1.0.0",
        "status": "running",
        "symptoms_available": len(symptoms_list)
    }

@app.get("/health")
async def health():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "symptoms_count": len(symptoms_list)
    }

@app.get("/api/sintomas")
async def get_sintomas():
    """Obtener lista de síntomas disponibles"""
    if not symptoms_list:
        raise HTTPException(status_code=500, detail="Lista de síntomas no disponible")

    # Convertir síntomas de snake_case a formato legible
    sintomas_formateados = []
    for symptom in symptoms_list:
        formatted = symptom.replace('_', ' ').title()
        sintomas_formateados.append({
            "id": symptom,
            "nombre": formatted,
            "descripcion": f"Síntoma: {formatted}"
        })

    return sintomas_formateados

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_disease(request: PredictionRequest):
    """Predecir enfermedad basada en síntomas"""

    if not request.consentimiento:
        raise HTTPException(status_code=400, detail="Debe aceptar el consentimiento")

    if not request.sintomas or len(request.sintomas) == 0:
        raise HTTPException(status_code=400, detail="Debe proporcionar al menos un síntoma")

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")

    try:
        # Crear DataFrame con los síntomas
        paciente_data = {}
        for symptom in symptoms_list:
            # Marcar como 1 si el síntoma está presente, 0 si no
            paciente_data[symptom] = 1 if symptom in request.sintomas else 0

        paciente_df = pd.DataFrame([paciente_data])

        # Si el modelo no está entrenado (modo de prueba), entrenar con datos de ejemplo
        if not hasattr(model, 'classes_'):
            logger.info("Entrenando modelo de prueba...")
            # Crear datos de ejemplo
            X_sample = pd.DataFrame([[0] * len(symptoms_list)] * 10, columns=symptoms_list)
            y_sample = ['Common Cold', 'Flu', 'Allergy', 'Gastritis', 'Migraine'] * 2
            model.fit(X_sample, y_sample)

        # Realizar predicción
        prediccion = model.predict(paciente_df)[0]
        probabilidades = model.predict_proba(paciente_df)[0]

        # Obtener las 3 enfermedades más probables
        top_indices = probabilidades.argsort()[-3:][::-1]

        predicciones = []
        for idx in top_indices:
            enfermedad = model.classes_[idx]
            probabilidad = float(probabilidades[idx])

            predicciones.append({
                "enfermedad": enfermedad,
                "probabilidad": probabilidad,
                "descripcion": f"Probabilidad de {enfermedad}: {probabilidad*100:.1f}%"
            })

        # Determinar urgencia
        max_prob = float(max(probabilidades))
        if max_prob > 0.8:
            urgencia = "alta"
        elif max_prob > 0.5:
            urgencia = "media"
        else:
            urgencia = "baja"

        # Generar recomendaciones
        recomendaciones = [
            "Consulte con un médico profesional para un diagnóstico preciso",
            "Mantenga un registro de sus síntomas y su evolución",
            "Beba suficiente agua y descanse adecuadamente"
        ]

        if urgencia == "alta":
            recomendaciones.insert(0, "⚠️ Se recomienda buscar atención médica urgente")

        return PredictionResponse(
            predicciones=predicciones,
            recomendaciones=recomendaciones,
            urgencia=urgencia,
            disclaimer="Esta predicción es solo una estimación basada en machine learning y NO sustituye el diagnóstico médico profesional. Siempre consulte con un profesional de la salud."
        )

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al realizar predicción: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
