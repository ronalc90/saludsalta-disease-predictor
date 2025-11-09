"""
Script para extraer la lista de síntomas del modelo entrenado
"""

import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("modelo_enfermedades.pkl")

# El modelo de Random Forest tiene guardadas las feature names
if hasattr(model, 'feature_names_in_'):
    symptoms = list(model.feature_names_in_)
    print(f"[OK] Se extrajeron {len(symptoms)} sintomas del modelo")
    print(f"Primeros 10 sintomas: {symptoms[:10]}")

    # Guardar la lista de síntomas
    joblib.dump(symptoms, "symptoms_list.pkl")
    print("[OK] Lista de sintomas guardada en symptoms_list.pkl")

    # También guardar en un archivo de texto para referencia
    with open("symptoms.txt", "w", encoding="utf-8") as f:
        for symptom in symptoms:
            f.write(f"{symptom}\n")
    print("[OK] Lista de sintomas guardada en symptoms.txt")
else:
    print("[ERROR] El modelo no tiene feature_names_in_")
    print("Intentando cargar desde el dataset de entrenamiento...")

    # Si el modelo no tiene los nombres, necesitaremos cargarlos de otro lugar
    # Por ahora usaremos una lista predefinida basada en el dataset común
    symptoms = [
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
        'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
        'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
        'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
        'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
        'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
        'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
        'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
        'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
        'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
        'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
        'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
        'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
        'spinning_movements', 'loss_of_balance', 'unsteadiness',
        'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
        'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases',
        'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
        'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
        'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
        'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
        'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
        'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
        'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
        'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
        'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
        'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ]
    joblib.dump(symptoms, "symptoms_list.pkl")
    print(f"[OK] Lista predefinida de {len(symptoms)} sintomas guardada")
