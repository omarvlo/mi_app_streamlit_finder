# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from openai import OpenAI

# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================
st.set_page_config(page_title="LLM + Analítica Descriptiva", page_icon="None")

st.title("📈 Analítica Descriptiva + LLM (Finder 2018–2021)")
st.markdown("""
Este módulo combina:

- **Analítica descriptiva local** (sin exponer datos sensibles)  
- **Interpretación de intención con LLM**  
- **Dashboard equivalente a Power BI con Altair**  
""")

# =========================================================
# 1) API KEY COMO CONTRASEÑA
# =========================================================
openai_api_key = st.text_input("🔑 Ingresa tu OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Agrega la API Key para activar el módulo LLM.", icon="⚠️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# =========================================================
# 2) FUNCIÓN SEGURA DEL LLM
# =========================================================
def ask_llm(prompt):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# =========================================================
# 3) CARGA DE DATOS FINDER (versión con uploader)
# =========================================================

st.subheader("📂 Cargar datasets de Finder (2018–2021)")

uploaded_files = st.file_uploader(
    "Sube los archivos CSV de Finder (4 archivos: 2018, 2019, 2020, 2021)",
    accept_multiple_files=True,
    type=["csv"]
)

def procesar_archivo(csv):
    df = pd.read_csv(
        csv,
        encoding="latin1",
        low_memory=False,
        na_values=["NADA","NULL","null","nan","NaN",""]
    )
    return df


@st.cache_data
def preparar_dataframe(dfs_dict):
    # Concatenar
    df = pd.concat(dfs_dict.values(), ignore_index=True)

    # Procesamiento estándar
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

    df["Familia"] = df["Familia"].astype(str).str.strip()
    df.loc[df["Familia"].str.lower().isin(["nan", "", "none"]), "Familia"] = np.nan

    return df


# ==========================================
# Validación + botón “Procesar”
# ==========================================

if uploaded_files:
    st.success(f"{len(uploaded_files)} archivo(s) cargado(s).")

    # Detectar por nombre a qué año pertenece cada archivo
    dfs_dict = {}

    for uf in uploaded_files:
        name = uf.name

        if "2018" in name:
            dfs_dict[2018] = procesar_archivo(uf)
        elif "2019" in name:
            dfs_dict[2019] = procesar_archivo(uf)
        elif "2020" in name:
            dfs_dict[2020] = procesar_archivo(uf)
        elif "2021" in name:
            dfs_dict[2021] = procesar_archivo(uf)
        else:
            st.error(f"No se reconoce el año en el archivo: {name}")

    # El usuario debe cargar los 4 archivos
    if len(dfs_dict) == 4:

        if st.button("Procesar datos"):
            df = preparar_dataframe(dfs_dict)
            st.success("Datos cargados y combinados correctamente 🎉")

            # Guardar en sesión para todo el módulo
            st.session_state["df_finder"] = df

    else:
        st.warning("Sube los 4 archivos: 2018, 2019, 2020 y 2021.")

# ==========================================
# Recuperar df para usar en todo el módulo
# ==========================================

if "df_finder" in st.session_state:
    df = st.session_state["df_finder"]
else:
    st.stop()


# =========================================================
# 4) INTÉRPRETE DE INTENCIÓN (LLM → JSON)
# =========================================================
def interpretar_intencion(texto, contexto):
    prompt = f"""
Eres un analista experto en ventas. Tu trabajo es interpretar consultas
en lenguaje natural y devolver SOLO un JSON válido.

Formato obligatorio:

{{
 "accion": "ventas_mes" | "ventas_año" | "promedio_familia" |
            "top_familias" | "resumen_mensual" |
            "recomendacion" | "explicacion" | "ayuda",
 "año": número | null,
 "mes": texto | null,
 "familia": texto | null,
 "k": número | null
}}

Reglas fuertes (OBEDECER SIEMPRE):

1. Si el usuario pregunta por "ventas en <mes> <año>" → accion = "ventas_mes".
2. Si el usuario menciona un mes (enero, feb, marzo, abril...) → es ventas_mes.
3. Si menciona solo un año → ventas_año.
4. Si menciona familia → promedio_familia.
5. Si menciona “top X familias” → top_familias.
6. Si pide estrategia, sugerencias → recomendacion.
7. Si usa referencias como “ese mes”, “lo anterior” → usa contexto:
   {contexto}
8. Si no puedes interpretar → accion = "ayuda".

Lista de meses reconocidos:
["enero","febrero","marzo","abril","mayo","junio",
 "julio","agosto","septiembre","octubre","noviembre","diciembre"]

Ejemplos:

Usuario: "¿Cuáles fueron las ventas en mayo del 2021?"
JSON:
{{"accion":"ventas_mes","año":2021,"mes":"mayo","familia":null,"k":null}}

Usuario: "Ventas en feb 2020"
JSON:
{{"accion":"ventas_mes","año":2020,"mes":"febrero","familia":null,"k":null}}

Usuario: "Top 3 familias 2020"
JSON:
{{"accion":"top_familias","año":2020,"mes":null,"familia":null,"k":3}}

Consulta del usuario:
"{texto}"

Devuelve SOLO el JSON, sin texto adicional.
"""
    raw = ask_llm(prompt)

    try:
        return json.loads(raw)
    except:
        return {"accion": "ayuda", "año": None, "mes": None, "familia": None, "k": None}


# =========================================================
# 5) FUNCIONES ANALÍTICAS
# =========================================================

def ventas_por_mes(df, año, mes):
    df_m = df[(df["Año"] == año) & (df["NombreMes"].str.lower() == mes.lower())]
    total = float(df_m["Cantidad"].sum())
    return {"año": año, "mes": mes, "total_unidades": total}

def ventas_por_año(df, año):
    df_y = df[df["Año"] == año]
    total = float(df_y["Cantidad"].sum())
    return {"año": año, "total_unidades": total}

def resumen_por_año(df, año):
    df_year = df[df["Año"] == año]
    return {
        "año": año,
        "total_unidades": float(df_year["Cantidad"].sum()),
        "promedio_unidades": float(df_year["Cantidad"].mean()),
        "mes_mayor_venta": df_year.groupby("NombreMes")["Cantidad"].sum().idxmax()
    }

def resumen_mensual(df, año):
    df_year = df[df["Año"] == año]
    tabla = df_year.groupby("NombreMes")["Cantidad"].sum().reset_index()
    tabla = tabla.sort_values("Cantidad", ascending=False)
    return {
        "año": año,
        "top_meses": tabla.to_dict(orient="records"),
        "mes_mayor_venta": tabla.iloc[0]["NombreMes"],
        "mes_menor_venta": tabla.iloc[-1]["NombreMes"],
    }

def promedio_familia(df, familia, año=None):
    df_fam = df[df["Familia"] == familia]
    if año:
        df_fam = df_fam[df_fam["Año"] == año]

    tabla = df_fam.groupby(["Año", "NombreMes"])["Cantidad"].mean().reset_index()
    return {
        "familia": familia,
        "resumen": tabla.to_dict(orient="records")
    }

def top_familias(df, año, k=5):
    df_year = df[df["Año"] == año]
    tabla = df_year.groupby("Familia")["Cantidad"].sum().reset_index()
    tabla = tabla.sort_values("Cantidad", ascending=False).head(k)
    return {
        "año": año,
        "top_familias": tabla.to_dict(orient="records")
    }


# =========================================================
# 6) CHAT + LLAMADAS A FUNCIONES
# =========================================================


st.subheader("💬 Chat con tu dataset")

# Inicializar sesión
if "contexto" not in st.session_state:
    st.session_state["contexto"] = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Entrada del usuario
if prompt := st.chat_input("Pregúntame sobre ventas…"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Interpretar intención con contexto
    intent = interpretar_intencion(prompt, st.session_state["contexto"])

    # Resolver
    if intent["accion"] == "ventas_mes":
        result = ventas_por_mes(df, intent["año"], intent["mes"])

    elif intent["accion"] == "ventas_año":
        result = ventas_por_año(df, intent["año"])

    elif intent["accion"] == "promedio_familia":
        result = promedio_familia(df, intent["familia"], intent["año"])

    elif intent["accion"] == "top_familias":
        result = top_familias(df, intent["año"], intent["k"])

    elif intent["accion"] == "resumen_mensual":
        result = resumen_mensual(df, intent["año"])

    elif intent["accion"] == "recomendacion":
        result = st.session_state["contexto"]  # reutilizar último contexto seguro

    elif intent["accion"] == "explicacion":
        result = {}

    else:
        result = {"mensaje": "Puedo responder sobre ventas por mes, año, familia, top meses, recomendaciones, etc."}

    # Actualizar contexto seguro
    nuevo_contexto = intent.copy()
    nuevo_contexto.update(result)
    st.session_state["contexto"] = nuevo_contexto

    # Mandar datos anonimizados al LLM
    final_prompt = f"""
    Eres un analista de datos.
    Responde EXCLUSIVAMENTE sobre los siguientes resultados ANONIMIZADOS:

    {json.dumps(result, ensure_ascii=False)}

    Reglas estrictas:
    - Describe e interpreta ÚNICAMENTE estos datos.
    - No conectes con mensajes anteriores.
    - No generes discurso largo.
    - No hagas recomendaciones a menos que el usuario las pida explícitamente.
    - No inventes valores ni tendencias que no se vean en este resultado.
    - Sé claro, breve y directo.

    Redacta una explicación concisa para el usuario.
    """

    respuesta = ask_llm(final_prompt)

    # Mostrar en interfaz
    with st.chat_message("assistant"):
        st.write(respuesta)

    # Guardar en historial
    st.session_state.messages.append({"role": "assistant", "content": respuesta})

    # =====================================================
# DASHBOARD DESCRIPTIVO (SE MUESTRA JUNTO AL CHATBOT)
# =====================================================

st.divider()
st.header("📊 Dashboard Descriptivo Finder 2018–2021")

st.markdown("""
Los siguientes gráficos se calculan **localmente**, usando los datos reales,
y complementan las respuestas del chatbot.
""")

# =====================================================
# 2) Gráfico de barras — Suma por Año y Mes
# =====================================================
st.subheader("📊 Suma de Cantidad por Año y Mes")

orden_meses = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

pivot_barras = (
    df.groupby(["Año", "NombreMes"], as_index=False)["Cantidad"].sum()
)
pivot_barras["NombreMes"] = pd.Categorical(
    pivot_barras["NombreMes"], categories=orden_meses, ordered=True
)

# Paleta Power BI
colores_powerbi = ["#00AEEF", "#1F4E79", "#F47B20", "#70AD47"]

chart_barras = (
    alt.Chart(pivot_barras)
    .mark_bar(size=12)
    .encode(
        x=alt.X("NombreMes:N", sort=orden_meses, title="Mes"),
        y=alt.Y("Cantidad:Q", title="Suma de Cantidad"),
        xOffset="Año:N",
        color=alt.Color(
            "Año:N",
            scale=alt.Scale(domain=[2018, 2019, 2020, 2021], range=colores_powerbi),
            title="Año"
        ),
        tooltip=["Año", "NombreMes", "Cantidad"]
    )
    .properties(height=420, title="Suma de Cantidad por Año y Mes")
)

st.altair_chart(chart_barras, use_container_width=True)


# =====================================================
# 3) Line plot — Evolución temporal
# =====================================================
st.subheader("📈 Evolución temporal de las ventas (2018–2021)")

chart_lineas = (
    alt.Chart(pivot_barras)
    .mark_line(point=True)
    .encode(
        x=alt.X("NombreMes:N", sort=orden_meses, title="Mes"),
        y="Cantidad:Q",
        color="Año:N",
        tooltip=["Año", "NombreMes", "Cantidad"]
    )
    .properties(title="Evolución mensual por año")
)

st.altair_chart(chart_lineas, use_container_width=True)


# =====================================================
# 4) Promedio por Familia
# =====================================================
st.subheader("🧩 Promedio de cantidad por Familia y Mes")

familias_disponibles = sorted(df["Familia"].dropna().unique())
familia_sel = st.selectbox("Selecciona una familia:", familias_disponibles)

pivot_familia = df.groupby(["Año","NombreMes","Familia"], as_index=False)["Cantidad"].mean()
df_filtro = pivot_familia[pivot_familia["Familia"] == familia_sel]

chart_familia = (
    alt.Chart(df_filtro)
    .mark_line(point=True)
    .encode(
        x=alt.X("NombreMes:N", sort=orden_meses),
        y=alt.Y("Cantidad:Q", title="Promedio de Cantidad"),
        color="Año:N",
        tooltip=["Año","NombreMes","Cantidad"]
    )
    .properties(title=f"Promedio de venta – Familia {familia_sel}")
)

st.altair_chart(chart_familia, use_container_width=True)


# =====================================================
# Reflexión
# =====================================================
st.markdown("""
### Reflexión
- Python y Altair permiten replicar dashboards corporativos con precisión.
- Este módulo prepara el terreno para **modelos predictivos** y **segmentación**.
""")


