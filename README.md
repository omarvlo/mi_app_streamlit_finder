# Plataforma de Analítica + LLM (Finder 2018–2021)

Aplicación interactiva construida con **Streamlit**, que combina:

- **Análisis descriptivo de ventas (2018–2021)**  
- **Interpretación automática de consulta con LLM (OpenAI)**  
- **Dashboards equivalentes a Power BI hechos con Altair**  
- **Privacidad garantizada mediante codificación Base64 y Secrets de Streamlit**

Este proyecto está diseñado para uso interno y análisis corporativo, manteniendo la seguridad de los datos sensibles fuera del repositorio.

---

## Características principales

### 1. Chat inteligente sobre ventas (LLM)
El usuario puede escribir preguntas como:

- *"¿Cuáles fueron las ventas en mayo de 2021?"*  
- *"Top 3 familias de 2020"*  
- *"Promedio mensual de la familia RESIDENCIAL"*  

El sistema:

1. **Interpreta la intención del usuario (NLU)**  
2. **Ejecuta funciones analíticas reales sobre los datos**  
3. **Produce una respuesta limpia basada solo en datos anonimizados**

Tecnología clave:

- `OpenAI GPT-4o-mini`  
- Motor de intención → reglas + JSON seguro  
- Prevención de alucinación mediante prompts restrictivos  

---

### 2. Dashboard descriptivo replicando capacidades de Power BI

Incluye:

- Suma de cantidad por año y mes  
- Evolución temporal de ventas  
- Promedios por familia de productos  
- Selección dinámica de familia  
- Visualizaciones con paleta Power BI  

Tecnología:

- `Altair`, `pandas`, `numpy`

---

### 3. Seguridad y privacidad de datos

Los archivos CSV reales **NO** están en GitHub.

En su lugar:

1. Fueron convertidos a Base64  
2. Se almacenan en **Streamlit Secrets**  
3. La app los decodifica en tiempo de ejecución  

Ventajas:

- ✔ Ningún dato sensible se expone en el repositorio  
- ✔ Sin rutas locales  
- ✔ Sin necesidad de servidores externos  

---

## 📁 Estructura del proyecto

