import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configuração da pagina inicial

st.set_page_config(
    page_title="Sistema Integrado de Obesidade",
    layout="wide"
)

st.title("**Sistema Integrado de Apoio Médico – Obesidade**")

st.write("""
Ferramenta com:

- Filtros clínicos
- Painel analítico
- Treinamento de modelo com PIPELINE
- Predição de nível de obesidade
""")

st.divider()

# Carga dos dados

df = pd.read_csv("obesity.csv")

target_column = "Obesity"

# Guardar cópia original para treino

df_original = df.copy()

# Barra lateral (Siderbar) - Filtros

st.sidebar.header("Filtros do Estudo")

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Categorias para uma unica escolha

gender_translation = {
    "All": "Todos",
    "Female": "Feminino",
    "Male": "Masculino"
}
gender_options_display = [gender_translation[g] for g in ["All"] + sorted(df["Gender"].unique().tolist())]
selected_display = st.sidebar.selectbox("**Gênero**", options=gender_options_display)
gender = [k for k, v in gender_translation.items() if v == selected_display][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------

family_history_translation = {
    "All": "Todos",
    "yes": "Sim",
    "no": "Não"
}
family_history_options_display = [family_history_translation[f] for f in ["All"] + sorted(df["family_history"].unique().tolist())]
selected_family_display = st.sidebar.selectbox(
    "**Algum membro da família sofreu ou sofre de excesso de peso?**",
    options=family_history_options_display
)
family_history = [k for k, v in family_history_translation.items() if v == selected_family_display][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------

favc_translation = {
    "All": "Todos",
    "yes": "Sim",
    "no": "Não"
}
favc_options_display = [favc_translation[f] for f in ["All"] + sorted(df["FAVC"].unique().tolist())]
selected_favc_display = st.sidebar.selectbox(
    "**Você come alimentos altamente calóricos com frequência? (FAVC)**",
    options=favc_options_display
)
favc = [k for k, v in favc_translation.items() if v == selected_favc_display][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------

caec_translation = {
    "All": "Todos",
    "Always": "Sempre",
    "Frequently": "Frequentemente",
    "Sometimes": "Às vezes",
    "no": "Não"
}
caec_options_display = [caec_translation[f] for f in ["All"] + sorted(df["CAEC"].unique().tolist())]

selected_caec_display = st.sidebar.selectbox(
    "**Você come alguma coisa entre as refeições? (CAEC)**",
    options=caec_options_display
)
caec = [k for k, v in caec_translation.items() if v == selected_caec_display][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------

smoke_translation = {
    "All": "Todos",
    "yes": "Sim",
    "no": "Não"
}
smoke_options_display = [smoke_translation[f] for f in ["All"] + sorted(df["SMOKE"].unique().tolist())]
selected_smoke_display = st.sidebar.selectbox(
    "**Você fuma? (SMOKE)**",
    options=smoke_options_display
)
smoke = [k for k, v in smoke_translation.items() if v == selected_smoke_display][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------

scc_translation = {
    "All": "Todos",
    "yes": "Sim",
    "no": "Não"
}
scc_options_display = [scc_translation[f] for f in ["All"] + sorted(df["SCC"].unique().tolist())]
selected_scc_display = st.sidebar.selectbox(
    "**Você monitora as calorias que ingere diariamente? (SCC)**",
    options=scc_options_display
)
scc = [k for k, v in scc_translation.items() if v == selected_scc_display][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------

calc_translation = {
    "All": "Todos",
    "Always": "Sempre",
    "Frequently": "Frequentemente",
    "Sometimes": "Às vezes",
    "no": "Não"
}
calc_options_display = [calc_translation[f] for f in ["All"] + sorted(df["CALC"].unique().tolist())]
selected_calc_display = st.sidebar.selectbox(
    "**Com que frequência você bebe álcool? (CALC)**",
    options=calc_options_display
)
calc = [k for k, v in calc_translation.items() if v == selected_calc_display][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------

mtrans_translation = {
    "All": "Todos",
    "Automobile": "Automóvel",
    "Bike": "Bicicleta",
    "Motorbike": "Motocicleta",
    "Public_Transportation": "Transporte Público",
    "Walking": "Caminhando"
}
mtrans_options_display = [mtrans_translation[f] for f in ["All"] + sorted(df["MTRANS"].unique().tolist())]
selected_mtrans_display = st.sidebar.selectbox(
    "**Qual meio de transporte você costuma usar? (MTRANS)**",
    options=mtrans_options_display
)
mtrans = [k for k, v in mtrans_translation.items() if v == selected_mtrans_display][0]

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Numericos com Slidebar para escolha

st.sidebar.markdown(
    "<p style='margin: 0 0 10px 0;'><strong>Idade</strong></p>",
    unsafe_allow_html=True
)

age = st.sidebar.slider(
    "",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max())),
    label_visibility="collapsed"
)

st.sidebar.markdown(
    "<p style='margin: 0 0 10px 0;'><strong>Altura em metros</strong></p>",
    unsafe_allow_html=True
)

height = st.sidebar.slider(
    "",
    float(df["Height"].min()),
    float(df["Height"].max()),
    (float(df["Height"].min()), float(df["Height"].max())),
    label_visibility="collapsed"
)

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Numericos porem com escolhas de inicio/até

st.sidebar.markdown("**Peso (kg)**")

min_w = float(df["Weight"].min())
max_w = float(df["Weight"].max())

col1, col2 = st.sidebar.columns(2)

with col1:
    weight_min = st.number_input(
        "De",
        min_value=min_w,
        max_value=max_w,
        value=min_w,
        step=0.5
    )

with col2:
    weight_max = st.number_input(
        "Até",
        min_value=weight_min,
        max_value=max_w,
        value=max_w,
        step=0.5
    )

weight = (weight_min, weight_max)

#-------------------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.markdown("**Consumo de vegetais (FCVC)**")

min_fcvc = float(df["FCVC"].min())
max_fcvc = float(df["FCVC"].max())

col1, col2 = st.sidebar.columns(2)

with col1:
    fcvc_min = st.number_input(
        "De",
        min_value=min_fcvc,
        max_value=max_fcvc,
        value=min_fcvc,
        step=1.0,
        key="fcvc_min"
    )

with col2:
    fcvc_max = st.number_input(
        "Até",
        min_value=fcvc_min,
        max_value=max_fcvc,
        value=max_fcvc,
        step=1.0,
        key="fcvc_max"
    )

fcvc = (fcvc_min, fcvc_max)

#-------------------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.markdown("**Refeições principais por dia (NCP)**")

min_ncp = float(df["NCP"].min())
max_ncp = float(df["NCP"].max())

col1, col2 = st.sidebar.columns(2)

with col1:
    ncp_min = st.number_input(
        "De",
        min_value=min_ncp,
        max_value=max_ncp,
        value=min_ncp,
        step=0.25,
        format="%.2f",
    )

with col2:
    ncp_max = st.number_input(
        "Até",
        min_value=ncp_min,
        max_value=max_ncp,
        value=max_ncp,
        step=0.25,
        format="%.2f",
    )

ncp = (ncp_min, ncp_max)

#-------------------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.markdown("**Consumo diário de água (CH2O)**")

min_ch2o = float(df["CH2O"].min())
max_ch2o = float(df["CH2O"].max())

col1, col2 = st.sidebar.columns(2)

with col1:
    ch2o_min = st.number_input(
        "De",
        min_value=min_ch2o,
        max_value=max_ch2o,
        value=min_ch2o,
        step=0.25,
        format="%.2f",
    )

with col2:
    ch2o_max = st.number_input(
        "Até",
        min_value=ch2o_min,
        max_value=max_ch2o,
        value=max_ch2o,
        step=0.25,
        format="%.2f",
    )

ch2o = (ch2o_min, ch2o_max)

#-------------------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.markdown("**Frequência de atividade física (FAF)**")

min_faf = float(df["FAF"].min())
max_faf = float(df["FAF"].max())

col1, col2 = st.sidebar.columns(2)

with col1:
    faf_min = st.number_input(
        "De",
        min_value=min_faf,
        max_value=max_faf,
        value=min_faf,
        step=0.25,
        format="%.2f",
        key="faf_min"
    )

with col2:
    faf_max = st.number_input(
        "Até",
        min_value=faf_min,
        max_value=max_faf,
        value=max_faf,
        step=0.25,
        format="%.2f",
        key="faf_max"
    )

faf = (faf_min, faf_max)

#-------------------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.markdown(
    "**Tempo de uso de dispositivos tecnológicos (TUE)**"
)

min_tue = float(df["TUE"].min())
max_tue = float(df["TUE"].max())

col1, col2 = st.sidebar.columns(2)

with col1:
    tue_min = st.number_input(
        "De",
        min_value=min_tue,
        max_value=max_tue,
        value=min_tue,
        step=0.25,
        format="%.2f",
        key="tue_min"
    )

with col2:
    tue_max = st.number_input(
        "Até",
        min_value=tue_min,
        max_value=max_tue,
        value=max_tue,
        step=0.25,
        format="%.2f",
        key="tue_max"
    )

tue = (tue_min, tue_max)

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Função dos filtros

def apply_select_filter(df, column, value):
    if value != "All":
        return df[df[column] == value]
    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Aplicação dos filtros

filtered_df = df.copy()

filtered_df = apply_select_filter(filtered_df, "Gender", gender)
filtered_df = apply_select_filter(filtered_df, "family_history", family_history)
filtered_df = apply_select_filter(filtered_df, "FAVC", favc)
filtered_df = apply_select_filter(filtered_df, "CAEC", caec)
filtered_df = apply_select_filter(filtered_df, "SMOKE", smoke)
filtered_df = apply_select_filter(filtered_df, "SCC", scc)
filtered_df = apply_select_filter(filtered_df, "CALC", calc)
filtered_df = apply_select_filter(filtered_df, "MTRANS", mtrans)

filtered_df = filtered_df[
    (filtered_df["Age"].between(age[0], age[1])) &
    (filtered_df["Height"].between(height[0], height[1])) &
    (filtered_df["Weight"].between(weight[0], weight[1])) &
    (filtered_df["FCVC"].between(fcvc[0], fcvc[1])) &
    (filtered_df["NCP"].between(ncp[0], ncp[1])) &
    (filtered_df["CH2O"].between(ch2o[0], ch2o[1])) &
    (filtered_df["FAF"].between(faf[0], faf[1])) &
    (filtered_df["TUE"].between(tue[0], tue[1]))
]

# Painel Analitico - Cada aplicação de filtro, atualiza

st.subheader("Painel Analítico")

st.write(f"Registros filtrados: {len(filtered_df)}")
st.dataframe(filtered_df.head(30), use_container_width=True)

if len(filtered_df) > 0:
    filtered_df["BMI"] = filtered_df["Weight"] / (filtered_df["Height"] ** 2)

    st.divider()

    st.metric("IMC Médio dos Filtrados", round(filtered_df["BMI"].mean(), 2))

    st.divider()

    # Tradução do grafico

    obesity_graphic_translation = {
        "Insufficient_Weight": "Peso insuficiente",
        "Normal_Weight": "Peso normal",
        "Overweight_Level_I": "Sobrepeso – Grau I",
        "Overweight_Level_II": "Sobrepeso – Grau II",
        "Overweight_Level_III": "Sobrepeso – Grau III",
        "Obesity_Type_I": "Obesidade – Tipo I",
        "Obesity_Type_II": "Obesidade – Tipo II",
        "Obesity_Type_III": "Obesidade – Tipo III",
    }

    st.subheader("Distribuição do Alvo")

    counts = filtered_df[target_column].value_counts()

    counts.index = [obesity_graphic_translation.get(idx, idx) for idx in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color="#4CAF50") 
    ax.set_title("Pacientes por Nível de Obesidade", fontsize=14)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Centraliza o gráfico

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.pyplot(fig)

    st.divider()

# Treinamento com o pipeline

st.subheader("Treinamento do Modelo Preditivo com Pipeline e Previsão de Obsesidade")

if st.button("Treinar Modelo com Pipeline"):

    X = df_original.drop(columns=[target_column])
    y = df_original[target_column]

    # Codificação
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    st.write("Acurácia:", round(accuracy_score(y_test, y_pred), 3))

    st.write(f"Acurácia em porcentagem: {round(accuracy_score(y_test, y_pred)*100, 2)}%")

    st.subheader("Relatório de Classificação")

    report = classification_report(y_test, y_pred, output_dict=True)

    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    joblib.dump(pipeline, "pipeline_obesity.pkl")
    joblib.dump(label_encoder, "labels.pkl")

    st.success("Pipeline treinado e salvo com sucesso!")

# Predição usando pipeline

st.divider()

# Tradução das colunas

column_translation = {
    "Age": "Idade",
    "Height": "Altura (m)",
    "Weight": "Peso (kg)",
    "BMI": "IMC",
    "Gender": "Gênero",
    "FCVC": "Consumo de vegetais",
    "NCP": "Número de refeições",
    "CH2O": "Consumo de água",
    "FAF": "Atividade física",
    "TUE": "Uso de tecnologia",
    "SMOKE": "Fumante",
    "CALC": "Consumo de álcool",
    "MTRANS": "Meio de transporte",
    "family_history": "Histórico familiar de sobrepeso",
    "FAVC": "Alimentos altamente calóricos",
    "CAEC": "Come entre refeições"
}

# Traduções das opções de seleção

gender_translation = {"Male": "Masculino", "Female": "Feminino"}
yes_no_translation = {"yes": "Sim", "no": "Não"}
caec_translation = {
    "Always": "Sempre",
    "Frequently": "Frequentemente",
    "Sometimes": "Às vezes",
    "no": "Não"
}
calc_translation = {
    "no": "Nunca",
    "Always": "Sempre",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente"
}
mtrans_translation = {
    "Automobile": "Carro",
    "Bike": "Bicicleta",
    "Motorbike": "Moto",
    "Public_Transportation": "Transporte Público",
    "Walking": "A Pé"
}
obesity_translation = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso – Grau I",
    "Overweight_Level_II": "Sobrepeso – Grau II",
    "Overweight_Level_III": "Sobrepeso – Grau III",
    "Obesity_Type_I": "Obesidade – Tipo I",
    "Obesity_Type_II": "Obesidade – Tipo II",
    "Obesity_Type_III": "Obesidade – Tipo III",
}

# ------------------------------

def sidebar_label(text):
    st.sidebar.markdown(
        f"<p style='margin: 4px 0 2px 0; font-size: 13px;'><strong>{text}</strong></p>",
        unsafe_allow_html=True
    )

# ------------------------------

try:
    pipeline = joblib.load("pipeline_obesity.pkl")
    label_encoder = joblib.load("labels.pkl")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Entrada para Predição")

    patient_inputs = {}

    for col in df_original.drop(columns=[target_column]).columns:

        sidebar_label(column_translation.get(col, col))

        if col == "Gender":
            options = [gender_translation[v] for v in sorted(df_original[col].unique())]
            selected = st.sidebar.selectbox("", options, key=f"pred_{col}", label_visibility="collapsed")
            patient_inputs[col] = [k for k, v in gender_translation.items() if v == selected][0]

        elif col in ["FAVC", "SMOKE", "SCC", "family_history"]:
            options = [yes_no_translation[v] for v in sorted(df_original[col].unique())]
            selected = st.sidebar.selectbox("", options, key=f"pred_{col}", label_visibility="collapsed")
            patient_inputs[col] = [k for k, v in yes_no_translation.items() if v == selected][0]

        elif col == "CAEC":
            options = [caec_translation[v] for v in sorted(df_original[col].unique())]
            selected = st.sidebar.selectbox("", options, key=f"pred_{col}", label_visibility="collapsed")
            patient_inputs[col] = [k for k, v in caec_translation.items() if v == selected][0]

        elif col == "CALC":
            options = [calc_translation[v] for v in sorted(df_original[col].unique())]
            selected = st.sidebar.selectbox("", options, key=f"pred_{col}", label_visibility="collapsed")
            patient_inputs[col] = [k for k, v in calc_translation.items() if v == selected][0]

        elif col == "MTRANS":
            options = [mtrans_translation[v] for v in sorted(df_original[col].unique())]
            selected = st.sidebar.selectbox("", options, key=f"pred_{col}", label_visibility="collapsed")
            patient_inputs[col] = [k for k, v in mtrans_translation.items() if v == selected][0]

        elif col == "Age":
            patient_inputs[col] = st.sidebar.number_input(
                "",
                min_value=int(df_original[col].min()),
                max_value=int(df_original[col].max()),
                value=int(df_original[col].mean()),
                step=1,
                key=f"pred_{col}",
                label_visibility="collapsed"
            )

        else:
            patient_inputs[col] = st.sidebar.number_input(
                "",
                value=float(df_original[col].mean()),
                key=f"pred_{col}",
                label_visibility="collapsed"
            )

    if st.button("Prever Obesidade do Paciente"):
        input_df = pd.DataFrame([patient_inputs])
        pred = pipeline.predict(input_df)
        result = label_encoder.inverse_transform(pred)
        translated_result = obesity_translation.get(result[0], result[0])
        st.success(f"Nível previsto: **{translated_result}**")

except Exception:
    st.info("Treine o modelo primeiro para habilitar a predição")