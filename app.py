import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Finder – Analítica + LLM",
    layout="wide"
)

# Portada de la aplicación
st.title("📊 Plataforma de Analítica y LLM – Finder 2018–2021")

st.markdown("""
Bienvenido a la plataforma interactiva de análisis de ventas de **Finder México**.

En el menú de la izquierda puedes acceder a:

- **Analítica descriptiva + Chat LLM**  
- **Dashboards interactivos**  
- **Módulos futuros (predicción, segmentación, anomalías)**  

Esta aplicación está construida para fines de análisis interno, manteniendo la privacidad 
de los datos mediante técnicas de codificación y secretos de Streamlit Cloud.
""")

st.info("Seleccione un módulo en el menú lateral.")
