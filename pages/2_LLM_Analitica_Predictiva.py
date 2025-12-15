# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json

from openai import OpenAI
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="LLM + Analítica Predictiva",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Analítica Predictiva + LLM (Finder 2018–2021)")
st.markdown("""
Este módulo integra:

- **Predicción trimestral multivariada (XGBoost)**
- **Análisis de estacionalidad y clustering**
- **Pipeline seguro LLM + Python**
""")

# =========================================================
# API KEY
# =========================================================
openai_api_key = st.text_input("🔑 Ingresa tu OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("Agrega la API Key para activar el módulo.", icon="⚠️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

def ask_llm(prompt: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def load_data():
    archivos = {
        2018: "FinderMX_data_2018.csv",
        2019: "FinderMX_data_2019.csv",
        2020: "FinderMX_data_2020.csv",
        2021: "FinderMX_data_2021.csv",
    }

    dfs = []
    for _, ruta in archivos.items():
        df_temp = pd.read_csv(
            ruta,
            encoding="latin1",
            na_values=["NADA", "NULL", "null", "NaN", "nan", ""],
            low_memory=False
        )
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)

    df["FechaMov"] = pd.to_datetime(df["FechaMov"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["FechaMov"])

    df["Año"] = df["FechaMov"].dt.year
    df["Mes"] = df["FechaMov"].dt.month

    meses = {
        1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
        7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",
        11:"Noviembre",12:"Diciembre"
    }
    df["NombreMes"] = df["Mes"].map(meses)

    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce")
    df = df.dropna(subset=["Cantidad"])

    df["Familia"] = df["Familia"].astype(str).str.strip()
    df.loc[df["Familia"].str.lower().isin(["nan", "", "none"]), "Familia"] = np.nan

    return df

df = load_data()


# =========================================================
# HELPERS DE MODELADO
# =========================================================
def add_calendar(df, col):
    d = pd.to_datetime(df[col])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["sin_year"] = np.sin(2*np.pi*d.dt.dayofyear/365.25)
    df["cos_year"] = np.cos(2*np.pi*d.dt.dayofyear/365.25)
    df["sin_quarter"] = np.sin(2*np.pi*df["quarter"]/4)
    df["cos_quarter"] = np.cos(2*np.pi*df["quarter"]/4)
    return df


# =========================================================
# PREDICCIÓN TRIMESTRAL
# =========================================================
def ejecutar_prediccion_trimestral(df, familia=None, año_prueba=2021):

    dfw = df.copy()
    if familia:
        dfw = dfw[dfw["Familia"] == familia]

    prod_col = "Familia"

    dfw = dfw[dfw["Año"].isin([2018, 2019, año_prueba])]
    dfw = add_calendar(dfw, "FechaMov")

    dfq = (
        dfw.groupby([prod_col, "Año", "quarter"])
           .agg({"Cantidad":"sum"})
           .reset_index()
    )

    dfq["lag1"] = dfq.groupby(prod_col)["Cantidad"].shift(1)
    dfq["lag2"] = dfq.groupby(prod_col)["Cantidad"].shift(2)

    dfq = dfq.dropna()

    train = dfq[dfq["Año"].isin([2018, 2019])]
    test  = dfq[dfq["Año"] == año_prueba]

    X_train = train[["lag1","lag2","quarter"]]
    y_train = train["Cantidad"]
    X_test  = test[["lag1","lag2","quarter"]]
    y_test  = test["Cantidad"]

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    r2   = float(r2_score(y_test, preds))

    return {
        "tipo": "prediccion_trimestral",
        "familia": familia,
        "año_prueba": año_prueba,
        "rmse": rmse,
        "r2": r2
    }


# =========================================================
# ESTACIONALIDAD + CLUSTERING
# =========================================================
def ejecutar_estacionalidad_clustering(df, k=4):

    pivot = (
        df.groupby(["Familia","Mes"])["Cantidad"]
          .sum()
          .groupby(level=0)
          .apply(lambda x: x / x.mean() * 100)
          .reset_index(name="indice")
    )

    mat = pivot.pivot(index="Familia", columns="Mes", values="indice").fillna(100)
    X = StandardScaler().fit_transform(mat.values)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    return {
        "tipo": "estacionalidad_clustering",
        "k": k,
        "metricas": {
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        }
    }


# =========================================================
# INTÉRPRETE DE INTENCIÓN
# =========================================================
def interpretar_intencion(texto):
    prompt = f"""
Devuelve SOLO JSON válido:

{{
 "accion": "prediccion_trimestral" | "estacionalidad_clustering" | "ayuda",
 "familia": texto | null,
 "año": número | null
}}

Consulta:
"{texto}"
"""
    try:
        return json.loads(ask_llm(prompt))
    except:
        return {"accion":"ayuda","familia":None,"año":None}


# =========================================================
# CHAT
# =========================================================
st.subheader("💬 Chat predictivo")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Pregunta sobre predicción o estacionalidad…"):

    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

    intent = interpretar_intencion(prompt)

    if intent["accion"] == "prediccion_trimestral":
        result = ejecutar_prediccion_trimestral(
            df,
            familia=intent["familia"],
            año_prueba=intent["año"] or 2021
        )

    elif intent["accion"] == "estacionalidad_clustering":
        result = ejecutar_estacionalidad_clustering(df)

    else:
        result = {"mensaje":"Puedo hacer predicción trimestral o estacionalidad."}

    final_prompt = f"""
Explica SOLO este resultado agregado:

{json.dumps(result, ensure_ascii=False)}

Reglas:
- No inventes datos
- 4–6 frases
- Lenguaje ejecutivo
"""
    respuesta = ask_llm(final_prompt)

    with st.chat_message("assistant"):
        st.write(respuesta)

    st.session_state.messages.append({"role":"assistant","content":respuesta})


# =========================================================
# DASHBOARD PREDICTIVO (BASE)
# =========================================================
st.divider()
st.header("📊 Contexto histórico (referencia)")

pivot = df.groupby(["Año","NombreMes"])["Cantidad"].sum().reset_index()

chart = (
    alt.Chart(pivot)
    .mark_line(point=True)
    .encode(
        x="NombreMes:N",
        y="Cantidad:Q",
        color="Año:N"
    )
    .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)
