
# Taller 3 — MLOps: FastAPI

Universidad EIA — Maestría en Analítica y Ciencia de Datos

## Descripción

Despliegue de un pipeline de Machine Learning mediante FastAPI para predecir costos de seguros médicos.

El pipeline integra:
- StandardScaler para variables numéricas.
- OneHotEncoder para variables categóricas.
- RandomForestRegressor como modelo predictivo.

## Archivos

- `train_pipeline.py`: entrenamiento y evaluación del modelo.
- `pipeline.joblib`: pipeline completo entrenado.
- `main.py`: aplicación FastAPI.
- `medical_insurance.csv`: conjunto de datos.
- `requirements.txt`: dependencias del proyecto.

## Instalación

```bash
conda create -n fastapi python=3.11 -y
conda activate fastapi
python -m pip install -r requirements.txt
```

## Entrenamiento

```bash
python train_pipeline.py
```

## Ejecutar FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /`: verifica el funcionamiento de la API.
- `GET /sum`: suma dos números.
- `POST /predict`: predice costos de seguros médicos.

Documentación interactiva:

`http://127.0.0.1:8000/docs`

## Ejemplo de predicción

```json
{
  "data_to_predict": [
    [23, "male", 34.4, 0, "no", "southwest"],
    [35, "female", 40, 1, "no", "northwest"]
  ]
}
```
