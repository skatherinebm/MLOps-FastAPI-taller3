
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Cargar los datos
df = pd.read_csv("medical_insurance.csv")

X = df.drop(columns=["charges"])
y = df["charges"]

# 2. Definir las variables
numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

# 3. Configurar el preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 4. Crear el pipeline completo
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    ))
])

# 5. Separar datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Entrenar y evaluar
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2:", round(r2_score(y_test, predictions), 4))

# 7. Guardar el pipeline entrenado
joblib.dump(pipeline, "pipeline.joblib")
print("Pipeline guardado correctamente.")
