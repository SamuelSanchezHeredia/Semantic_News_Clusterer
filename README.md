# 🗞️ Semantic News Clusterer

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BERT](https://img.shields.io/badge/BERT-Sentence_Transformers-orange.svg)](https://www.sbert.net/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

## 📖 Descripción

**Semantic News Clusterer** es un sistema avanzado de clustering semántico que agrupa automáticamente noticias similares utilizando técnicas de NLP de última generación. A diferencia de los métodos tradicionales, este proyecto **no requiere etiquetas previas** y agrupa los textos basándose en su **significado semántico**, no solo en palabras clave.

### 🎯 Objetivo

Desarrollar un pipeline completo para:
- 🔍 Descubrir automáticamente temas en grandes volúmenes de noticias
- 📊 Identificar patrones y tendencias sin supervisión humana
- 🎨 Visualizar clusters de manera interactiva
- 📝 Interpretar y entender qué representa cada cluster
- 🚀 Aplicación web interactiva con Streamlit para clasificar noticias en tiempo real

---

## 📑 Tabla de Contenidos

1. [Stack Tecnológico](#-stack-tecnológico)
2. [Inicio Rápido](#-inicio-rápido)
3. [Instalación Completa](#-instalación-completa)
4. [Aplicación Streamlit](#-aplicación-streamlit)
5. [Notebook Jupyter](#-notebook-jupyter)
6. [Integración del Modelo](#-integración-del-modelo-real)
7. [Dataset](#-dataset)
8. [Metodología](#-metodología)
9. [Configuración](#-configuración-personalizada)
10. [Solución de Problemas](#-solución-de-problemas)
11. [Referencias](#-referencias-y-recursos)

---

## 🛠️ Stack Tecnológico

### Componentes Principales

| Tecnología | Propósito | Versión |
|------------|-----------|---------|
| **BERT** (Sentence Transformers) | Embeddings semánticos | 5.2.2+ |
| **UMAP** | Reducción de dimensionalidad | 0.5.11+ |
| **HDBSCAN** | Clustering jerárquico | 0.8.41+ |
| **Plotly** | Visualizaciones interactivas | 6.5.2+ |
| **Streamlit** | Aplicación web interactiva | 1.31.0+ |
| **Pandas** | Manipulación de datos | 2.3.3+ |
| **Jupyter** | Entorno interactivo | 1.1.1+ |

### Pipeline del Sistema

```
📥 Descarga → 🧹 Limpieza → 🤖 BERT (384D) → 🔽 UMAP (5D) → 🔍 HDBSCAN → 📊 Visualización → 📝 Interpretación
```

---

## ⚡ Inicio Rápido

### 1️⃣ Activar el entorno virtual

```bash
cd Semantic_News_Clusterer
source .venv/bin/activate  # En macOS/Linux
# .venv\Scripts\activate   # En Windows
```

### 2️⃣ Elegir tu método

#### Opción A: 🌐 Aplicación Web Streamlit (Recomendado)

```bash
# Usar el script unificado
./run.sh streamlit

# O manualmente
streamlit run app_streamlit.py
```

**Características:**
- 📝 Clasificar noticias individuales en tiempo real
- 📊 Explorar clusters temáticos
- 🔬 Análisis por lotes (múltiples noticias)
- 🎨 Visualizaciones interactivas

**Acceder en:** http://localhost:8501

#### Opción B: 📓 Jupyter Notebook (Análisis completo)

```bash
# Usar el script unificado
./run.sh notebook

# O manualmente
jupyter notebook
```

Abre **`clustering_noticias_bert_hdbscan.ipynb`** y ejecuta:
- **Cell → Run All** (para ejecutar todo)
- O **Shift + Enter** celda por celda

### 3️⃣ Flujo Completo (Entrenar + Streamlit)

```bash
# Script unificado que hace todo
./run.sh full
```

Este comando:
1. Verifica dependencias
2. Detecta si existe modelo entrenado
3. Te pregunta si quieres entrenar o usar existente
4. Ejecuta Streamlit con el modelo

⏱️ **Tiempo estimado**: 
- App Streamlit: ~2 minutos (inicio)
- Notebook completo: ~10-15 minutos con 10,000 noticias

---

## 🚀 Instalación Completa

### Pre-requisitos

- Python 3.8 o superior
- 8GB de RAM mínimo
- Conexión a Internet (para descargar datos y modelos)

### Instalación desde cero

```bash
# 1. Clonar o descargar el repositorio
cd Semantic_News_Clusterer

# 2. Crear entorno virtual
python3 -m venv .venv

# 3. Activar entorno virtual
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 4. Actualizar pip
pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt

# 6. Verificar instalación
python verificar_entorno.py

# 7. Ejecutar la app
./run.sh streamlit
```

---

## 🌐 Aplicación Streamlit

### 🎯 Características

La aplicación web incluye 3 tabs principales:

#### Tab 1: 📝 Clasificar Noticia Individual
- Formulario para introducir titular y descripción
- Botones de ejemplo rápido (Política, Entretenimiento, Tecnología)
- Muestra:
  - Cluster asignado con nombre descriptivo
  - Porcentaje de confianza
  - Términos clave detectados
  - Métricas (palabras, cluster ID)
- Configuración avanzada (ver embeddings, preprocesamiento)

#### Tab 2: 📊 Explorar Clusters
- Visualiza todos los clusters temáticos disponibles
- Gráfico de distribución interactivo
- Detalles expandibles por cluster
- Información del modelo (si está entrenado)

#### Tab 3: 🔬 Análisis por Lotes
- **Opción 1**: Pegar múltiples noticias (formato: `Titular | Descripción`)
- **Opción 2**: Cargar archivo CSV
- Procesa todas las noticias a la vez
- Muestra tabla de resultados
- Gráfico de distribución (pie chart)
- Botón para descargar resultados en CSV

### 🚀 Ejecutar la Aplicación

```bash
# Método 1: Script unificado
./run.sh streamlit

# Método 2: Directamente
streamlit run app_streamlit.py

# Acceder en: http://localhost:8501
```

### 💡 Ejemplos de Uso

#### Ejemplo 1: Noticia Política
```
Titular: Biden announces new climate policy
Descripción: The president unveiled sweeping climate change initiatives
```

#### Ejemplo 2: Noticia Tecnológica
```
Titular: Apple unveils new iPhone with AI features
Descripción: The tech giant announced revolutionary artificial intelligence capabilities
```

#### Ejemplo 3: Análisis Batch
Pega esto en el Tab 3:
```
Trump announces policy | New immigration rules
Apple releases iPhone | New AI features included
Lakers win championship | Basketball team claims title
```

### 🛠️ Tecnologías de la App

- **Streamlit**: Framework de aplicación web
- **BERT** (all-MiniLM-L6-v2): Modelo de embeddings semánticos
- **scikit-learn**: Cálculo de similitud de coseno
- **Plotly**: Visualizaciones interactivas
- **Pandas/NumPy**: Procesamiento de datos

---

## 📓 Notebook Jupyter

### 📖 Estructura del Notebook

El notebook está organizado en **11 secciones**:

1. **Importación de Librerías**: Setup inicial
2. **Carga de Datos**: Descarga desde Kaggle
3. **Preparación**: Concatenación de campos
4. **Preprocesamiento**: Limpieza de texto
5. **Embeddings BERT**: Vectorización semántica
6. **Reducción UMAP**: 5D para clustering, 2D para visualización
7. **Clustering HDBSCAN**: Identificación automática de grupos
8. **Visualización**: Gráficos interactivos
9. **Interpretación**: Palabras clave y ejemplos
10. **Análisis de Calidad**: Métricas y evaluación
11. **Exportación**: Guardar modelo para Streamlit

### 🚀 Ejecutar el Notebook

```bash
# Método 1: Script unificado
./run.sh notebook

# Método 2: Directamente
jupyter notebook

# Abrir: clustering_noticias_bert_hdbscan.ipynb
# Ejecutar: Cell → Run All
```

---

## 🔗 Integración del Modelo Real

### ¿Cómo funciona?

1. **Notebook** entrena el modelo con 10,000 noticias y guarda centroides en `model_data.pkl`
2. **Streamlit** carga automáticamente el modelo real o usa demo si no existe

### 🚀 Usar el Modelo Real (3 Pasos)

#### Paso 1: Entrenar Modelo en el Notebook

```bash
./run.sh notebook
# O directamente: jupyter notebook
```

1. Abre `clustering_noticias_bert_hdbscan.ipynb`
2. **Ejecuta TODAS las celdas** (Cell → Run All)
3. **Importante**: La última sección guarda el modelo automáticamente

**Verás al final:**
```
💾 GUARDANDO MODELO PARA LA APLICACIÓN STREAMLIT
✓ Centroides calculados para X clusters
✓ Modelo guardado en: model_data.pkl
✓ Tamaño del archivo: XX KB

✅ MODELO LISTO PARA USAR EN STREAMLIT
```

#### Paso 2: Verificar Archivo

```bash
ls -lh model_data.pkl
# Debe existir y tener tamaño > 0
```

#### Paso 3: Ejecutar Streamlit

```bash
./run.sh streamlit
```

**Verás en la app:**
```
✅ Modelo REAL cargado correctamente

📊 Ver información del modelo entrenado ▼
   Clusters: X
   Noticias entrenadas: 10,000
   Dimensión: 5D
```

### 🔍 Contenido de model_data.pkl

```python
{
    'centroids': {
        0: array([...]),  # Centroide del cluster 0 (384D - BERT original)
        1: array([...]),  # Centroide del cluster 1
        # ... más clusters
    },
    'cluster_names': {
        0: "Trump & President (White)",
        1: "Movie & Film (Star)",
        # ... (nombres generados automáticamente)
    },
    'model_name': 'all-MiniLM-L6-v2',
    'n_clusters': 6,
    'n_samples': 10000,
    'embedding_dimension': 384,  # BERT genera 384D
    'centroid_dimension': 384    # Centroides también en 384D
}
```

### 🎯 Características Modelo Real

| Aspecto | Modelo Real |
|---------|-------------|
| **Origen** | Entrenado con 10,000 noticias |
| **Centroides** | `mean(cluster_embeddings)` |
| **Clusters** | Los que HDBSCAN encontró |
| **Nombres** | Generados de términos reales |
| **Precisión**  | ✅ Alta (~80%) |
| **Confianza** | >70% típicamente |

---

## 📊 Dataset

### News Category Dataset

- **Fuente**: [Kaggle - News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Contenido**: ~200,000 noticias de HuffPost con categorías
- **Formato**: JSON Lines
- **Descarga**: Automática mediante `kagglehub`
- **Campos principales**:
  - `headline`: Título de la noticia
  - `short_description`: Descripción breve
  - `category`: Categoría original (solo para validación)

**El dataset se descarga automáticamente al ejecutar el notebook.** No requiere configuración manual.

---

## 🔬 Metodología

### Pipeline Detallado

#### 1. **Carga de Datos**
- Descarga automática desde Kaggle
- Lectura de archivos JSON Lines
- Concatenación de `headline` + `short_description`

#### 2. **Preprocesamiento**
```python
- Conversión a minúsculas
- Eliminación de URLs y caracteres especiales
- Preservación de estructura semántica para BERT
- Filtrado de textos muy cortos
```

#### 3. **Generación de Embeddings**
```python
Modelo: 'all-MiniLM-L6-v2'
- Dimensiones: 384
- Velocidad: ~1000 textos/segundo
- Optimizado para similitud semántica
```

#### 4. **Reducción Dimensional (UMAP)**
```python
- De 384D → 5D (para clustering)
- De 384D → 2D (para visualización)
- Preserva estructura local y global
```

#### 5. **Clustering (HDBSCAN)**
```python
- Identifica clusters automáticamente
- Separa outliers (ruido)
- Proporciona probabilidades de asignación
```

#### 6. **Generación de Nombres Descriptivos**
```python
- Extrae los 5 términos más frecuentes por cluster
- Genera nombres automáticos como "Trump & President (Election)"
- Filtra stopwords
```

#### 7. **Visualización e Interpretación**
```python
- Gráficos interactivos 2D (Plotly)
- Top 5 palabras clave por cluster
- Títulos representativos
- Análisis de coherencia
```

---

## ⚙️ Configuración Personalizada

### Ajustar Número de Noticias

```python
# En el notebook:

# Prueba rápida (2 minutos)
df = load_news_dataset(sample_size=1000)

# Análisis medio (10-15 minutos) - RECOMENDADO
df = load_news_dataset(sample_size=10000)

# Análisis completo (~200k noticias, 1-2 horas)
df = load_news_dataset(sample_size=None)
```

### Ajustar Clustering (HDBSCAN)

#### Más clusters pequeños (análisis detallado):
```python
labels, clusterer = perform_clustering(
    embeddings_5d,
    min_cluster_size=30,   # ⬇️ Reducir
    min_samples=5          # ⬇️ Reducir
)
```

#### Menos clusters grandes (visión general):
```python
labels, clusterer = perform_clustering(
    embeddings_5d,
    min_cluster_size=100,  # ⬆️ Aumentar
    min_samples=20         # ⬆️ Aumentar
)
```

### Parámetros UMAP

```python
# Estructura local (muchos clusters pequeños)
embeddings_5d = reduce_dimensions(
    embeddings, n_components=5, n_neighbors=5
)

# Balance - RECOMENDADO
embeddings_5d = reduce_dimensions(
    embeddings, n_components=5, n_neighbors=15
)

# Estructura global (pocos clusters grandes)
embeddings_5d = reduce_dimensions(
    embeddings, n_components=5, n_neighbors=30
)
```

---

## 📈 Resultados Esperados

### Con 10,000 noticias (configuración por defecto)

| Métrica | Valor Típico |
|---------|--------------|
| **Clusters identificados** | 15-25 |
| **Cobertura** | 85-90% en clusters |
| **Ruido** | 10-15% |
| **Tiempo de ejecución** | 10-15 minutos |
| **Confianza media** | >70% |

### Métricas de Calidad

| Métrica | ✅ Ideal | ⚠️ Aceptable | ❌ Problemático |
|---------|----------|--------------|-----------------|
| **Clusters** | 15-30 | 10-50 | <5 o >100 |
| **Ruido %** | 5-15% | 15-25% | >30% |
| **Confianza** | >0.75 | 0.6-0.75 | <0.6 |

---

## 🎛️ Script Unificado (run.sh)

El proyecto incluye un script unificado que maneja todas las operaciones:

```bash
# Ver ayuda
./run.sh help

# Ejecutar Streamlit
./run.sh streamlit

# Ejecutar Jupyter Notebook
./run.sh notebook

# Flujo completo (entrenar + streamlit)
./run.sh full

# Verificar instalación
./run.sh check

# Limpiar archivos temporales
./run.sh clean
```

---

## 🎓 Casos de Uso

### 📰 Análisis de Medios
- Identificar temas recurrentes en cobertura noticiosa
- Detectar sesgos informativos
- Comparar diferentes fuentes

### 📈 Vigilancia de Tendencias
- Descubrir temas emergentes
- Monitorear evolución de noticias
- Alertas de nuevos temas

### 🗂️ Organización de Contenido
- Agrupar artículos similares
- Sistemas de recomendación
- Deduplicación de noticias

### 🔬 Research Académico
- Análisis de corpus de texto
- Estudios de comunicación
- Análisis de sentimiento por cluster

---

## 📚 Referencias y Recursos

### Documentación Técnica

- [Sentence Transformers](https://www.sbert.net/) - Modelos BERT
- [UMAP Documentation](https://umap-learn.readthedocs.io/) - Reducción dimensional
- [HDBSCAN Guide](https://hdbscan.readthedocs.io/) - Clustering
- [Streamlit Docs](https://docs.streamlit.io/) - Framework web
- [News Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) - Datos

### Papers Relevantes

- Reimers & Gurevych (2019): "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- McInnes et al. (2018): "UMAP: Uniform Manifold Approximation and Projection"
- Campello et al. (2013): "Density-Based Clustering Based on Hierarchical Density Estimates"

---

## 📂 Estructura del Proyecto

```
Semantic_News_Clusterer/
├── 📓 clustering_noticias_bert_hdbscan.ipynb  # Notebook principal
├── 🌐 app_streamlit.py                        # Aplicación web
├── 🔧 run.sh                                  # Script unificado
├── ✅ verificar_entorno.py                    # Verificador
├── 📋 requirements.txt                        # Dependencias
├── 💾 model_data.pkl                          # Modelo entrenado (se genera)
├── 📁 .venv/                                  # Entorno virtual
└── 📖 README.md                               # Este archivo
```

---

## 👤 Autor y Contribuciones

**Samuel Sanchez Heredia**

---

## 📄 Licencia

Este proyecto está disponible bajo licencia MIT para uso educativo y de investigación.

---

## ⭐ Si este proyecto te fue útil

- Dale una ⭐ en GitHub
- Compártelo con otros
- Contribuye con mejoras
- Úsalo como base para tus proyectos

---

<div align="center">

**🗞️ Semantic News Clusterer**

Desarrollado usando Python, BERT, UMAP, HDBSCAN y Streamlit

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red?logo=Streamlit)](https://streamlit.io/)

---

**Última actualización**: Febrero 2026

**Versión**: 2.0

</div>

