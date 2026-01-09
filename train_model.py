import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("obesity.csv")

target_column = "Obesity"

X = df.drop(columns=[target_column])
y = df[target_column]


# Codificar variáveis categóricas
encoders = {}

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Escalar dados
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# Codificar variável alvo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


joblib.dump(model, "obesity_model.pkl")
joblib.dump(label_encoder, "labels.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\nArquivos gerados:")
print("- obesity_model.pkl")
print("- labels.pkl")
print("- scaler.pkl")
print("- encoders.pkl")
