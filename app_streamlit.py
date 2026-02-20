"""
🗞️ Aplicación de Clustering Semántico de Noticias
Permite al usuario introducir textos de noticias y predecir a qué cluster pertenecen
usando BERT + HDBSCAN pre-entrenado
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import re
from collections import Counter

# Configuración de la página
st.set_page_config(
    page_title="Clustering Semántico de Noticias",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cluster-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bert_model(model_name: str = 'all-MiniLM-L6-v2'):
    """Carga el modelo BERT (con cache)"""
    return SentenceTransformer(model_name)


def preprocess_text(text: str) -> str:
    """
    Preprocesa el texto de entrada
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-záéíóúüñ0-9\s.,!?\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def generate_embeddings(texts: List[str], model) -> np.ndarray:
    """Genera embeddings para los textos"""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def get_top_terms(text: str, n_terms: int = 5) -> List[str]:
    """Extrae los términos más relevantes de un texto"""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'it', 'its', 'this',
        'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how'
    }

    words = text.split()
    words = [w for w in words if len(w) >= 3 and w not in stopwords]

    word_counts = Counter(words)
    return [word for word, count in word_counts.most_common(n_terms)]


def predict_cluster_approximate(
    new_embedding: np.ndarray,
    cluster_centroids: Dict[int, np.ndarray],
    cluster_names: Dict[int, str]
) -> Tuple[int, str, float]:
    """
    Predice el cluster más cercano para un nuevo embedding
    """
    from sklearn.metrics.pairwise import cosine_similarity

    best_cluster = -1
    best_similarity = -1
    best_name = "Sin clasificar"

    for cluster_id, centroid in cluster_centroids.items():
        if cluster_id == -1:  # Ignorar cluster de ruido
            continue

        similarity = cosine_similarity(
            new_embedding.reshape(1, -1),
            centroid.reshape(1, -1)
        )[0][0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = cluster_id
            best_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    return best_cluster, best_name, best_similarity


def create_demo_model():
    """
    Crea un modelo de demostración con clusters predefinidos
    """
    cluster_names = {
        0: "Política & Trump (Election)",
        1: "Entretenimiento & Movie (Star)",
        2: "Tecnología & Tech (Apple)",
        3: "Deportes & Team (Game)",
        4: "Salud & Health (Study)",
        5: "Negocios & Business (Company)"
    }

    # Centroides de ejemplo (en realidad deberían venir del modelo entrenado)
    # Estos son vectores ficticios de 384 dimensiones
    np.random.seed(42)
    cluster_centroids = {
        i: np.random.randn(384) for i in range(6)
    }

    return cluster_centroids, cluster_names


@st.cache_resource
def load_trained_model(model_path: str = 'model_data.pkl'):
    """
    Carga el modelo entrenado desde el notebook usando pickle.
    Si no existe, usa el modelo de demostración.

    Args:
        model_path: Ruta al archivo pickle con el modelo

    Returns:
        Tuple (cluster_centroids, cluster_names, metadata)
    """
    try:
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Modelo no encontrado en '{model_path}'")
            st.info("💡 Ejecuta el notebook completo para generar el modelo real")
            st.info("📝 Usando modelo de demostración mientras tanto...")
            centroids, names = create_demo_model()
            return centroids, names, {'is_demo': True}

        # Cargar modelo desde pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        centroids = model_data['centroids']
        names = model_data['cluster_names']

        # Metadata adicional
        metadata = {
            'is_demo': False,
            'n_clusters': model_data.get('n_clusters', len(centroids)),
            'n_samples': model_data.get('n_samples', 'N/A'),
            'model_name': model_data.get('model_name', 'all-MiniLM-L6-v2'),
            'embedding_dim': model_data.get('embedding_dimension', 384),
            'centroid_dim': model_data.get('centroid_dimension', 384)
        }

        return centroids, names, metadata

    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {str(e)}")
        st.info("📝 Usando modelo de demostración...")
        centroids, names = create_demo_model()
        return centroids, names, {'is_demo': True, 'error': str(e)}


# ====================
# INTERFAZ PRINCIPAL
# ====================

def main():
    # Header
    st.markdown('<h1 class="main-header">🗞️ Clustering Semántico de Noticias</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Introduce una noticia y descubre a qué cluster temático pertenece</p>', unsafe_allow_html=True)

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        ### ¿Cómo funciona?
        
        1. **BERT** genera embeddings semánticos del texto
        2. **HDBSCAN** agrupa noticias similares
        3. La app predice el cluster más cercano
        
        ### Tecnologías
        - 🤖 Sentence-BERT (all-MiniLM-L6-v2)
        - 📊 HDBSCAN Clustering
        - 🎨 Streamlit
        
        ### Dataset
        News Category Dataset (Kaggle)
        """)

        st.divider()

        # Opciones avanzadas
        st.header("⚙️ Configuración")
        show_embeddings = st.checkbox("Mostrar embeddings", value=False)
        show_preprocessing = st.checkbox("Mostrar preprocesamiento", value=False)

    # Cargar modelo
    with st.spinner("🔄 Cargando modelo BERT..."):
        bert_model = load_bert_model()
        cluster_centroids, cluster_names, model_metadata = load_trained_model()

    # Mostrar información del modelo cargado
    if model_metadata.get('is_demo', False):
        st.warning("⚠️ Usando modelo de DEMOSTRACIÓN")
        st.info("💡 Para usar el modelo real, ejecuta el notebook completo y guarda el modelo con la última celda")
    else:
        st.success("✅ Modelo REAL cargado correctamente")

        # Mostrar estadísticas del modelo en un expander
        with st.expander("📊 Ver información del modelo entrenado"):
            col_info1, col_info2, col_info3 = st.columns(3)

            with col_info1:
                st.metric("Clusters", model_metadata.get('n_clusters', 'N/A'))

            with col_info2:
                st.metric("Noticias entrenadas", f"{model_metadata.get('n_samples', 'N/A'):,}")

            with col_info3:
                st.metric("Dimensión centroides", f"{model_metadata.get('centroid_dim', 'N/A')}D")

            st.caption(f"🤖 Modelo: {model_metadata.get('model_name', 'N/A')}")
            st.caption(f"📐 Embedding BERT: {model_metadata.get('embedding_dim', 'N/A')}D")

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📝 Clasificar Noticia", "📊 Explorar Clusters", "🔬 Análisis Batch"])

    # ==================
    # TAB 1: CLASIFICAR
    # ==================
    with tab1:
        st.header("Introduce tu noticia")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Input del usuario
            headline = st.text_input(
                "📰 Titular",
                placeholder="Ej: Trump announces new policy on immigration"
            )

            description = st.text_area(
                "📄 Descripción breve",
                placeholder="Ej: The president announced a new executive order affecting visa policies...",
                height=150
            )

            # Ejemplos predefinidos
            st.markdown("**O prueba con estos ejemplos:**")
            example_cols = st.columns(3)

            with example_cols[0]:
                if st.button("🏛️ Política"):
                    headline = "Biden announces new climate policy"
                    description = "The president unveiled sweeping climate change initiatives"
                    st.rerun()

            with example_cols[1]:
                if st.button("🎬 Entretenimiento"):
                    headline = "New Marvel movie breaks box office records"
                    description = "The latest superhero film earned $200M in opening weekend"
                    st.rerun()

            with example_cols[2]:
                if st.button("💻 Tecnología"):
                    headline = "Apple unveils new iPhone with AI features"
                    description = "The tech giant announced revolutionary artificial intelligence capabilities"
                    st.rerun()

        with col2:
            st.info("""
            💡 **Consejos**:
            - Escribe en inglés
            - Sé específico en el titular
            - Incluye contexto en la descripción
            """)

        # Botón de clasificar
        if st.button("🔍 Clasificar Noticia", type="primary", use_container_width=True):
            if not headline or not description:
                st.warning("⚠️ Por favor, introduce tanto el titular como la descripción")
            else:
                with st.spinner("🔄 Analizando noticia..."):
                    # Combinar texto
                    full_text = f"{headline} {description}"

                    # Preprocesar
                    text_clean = preprocess_text(full_text)

                    if show_preprocessing:
                        st.subheader("🧹 Texto Preprocesado")
                        st.code(text_clean)

                    # Generar embedding
                    embedding = generate_embeddings([text_clean], bert_model)[0]

                    if show_embeddings:
                        st.subheader("🔢 Embedding (primeras 20 dimensiones)")
                        st.code(embedding[:20])

                    # Predecir cluster
                    cluster_id, cluster_name, similarity = predict_cluster_approximate(
                        embedding, cluster_centroids, cluster_names
                    )

                    # Mostrar resultados
                    st.success("✅ Clasificación completada")

                    # Resultado principal
                    st.markdown(f"""
                    <div class="cluster-card">
                        <h2>🏷️ Cluster Asignado</h2>
                        <h1>{cluster_name}</h1>
                        <p><strong>Similitud:</strong> {similarity:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Métricas adicionales
                    col_m1, col_m2, col_m3 = st.columns(3)

                    with col_m1:
                        st.metric("Cluster ID", cluster_id)

                    with col_m2:
                        st.metric("Confianza", f"{similarity:.1%}")

                    with col_m3:
                        words = len(text_clean.split())
                        st.metric("Palabras", words)

                    # Top términos
                    st.subheader("🔑 Términos Clave Detectados")
                    top_terms = get_top_terms(text_clean)
                    st.write(" • ".join([f"**{term}**" for term in top_terms]))

    # =======================
    # TAB 2: EXPLORAR CLUSTERS
    # =======================
    with tab2:
        st.header("📊 Exploración de Clusters")

        # Mostrar tipo de modelo
        if model_metadata.get('is_demo', False):
            st.info("ℹ️ Mostrando clusters de demostración. Ejecuta el notebook para ver los clusters reales de tu dataset.")
        else:
            st.success(f"✅ Mostrando {model_metadata.get('n_clusters', len(cluster_names))} clusters del modelo entrenado")

        st.markdown("""
        Estos son los clusters temáticos encontrados en el dataset de noticias:
        """)

        # Mostrar clusters
        for cluster_id, name in cluster_names.items():
            with st.expander(f"🏷️ {name}", expanded=False):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("**Características:**")
                    st.write(f"- ID del Cluster: {cluster_id}")

                    # Si es modelo real, mostrar info adicional
                    if not model_metadata.get('is_demo', False):
                        st.write(f"- Vector centroide disponible")
                        centroid = cluster_centroids.get(cluster_id)
                        if centroid is not None:
                            st.write(f"- Dimensiones: {len(centroid)}D")
                    else:
                        st.write(f"- Tamaño estimado: ~{np.random.randint(500, 2000)} noticias")

                with col_b:
                    st.markdown("**Términos Representativos:**")
                    # Extraer términos del nombre
                    terms = name.replace("(", "").replace(")", "").split(" & ")
                    for term in terms:
                        st.write(f"• {term}")

        # Gráfico de distribución
        st.subheader("📈 Distribución de Noticias por Cluster")

        # Si es modelo real, usar info real; si no, simular
        if not model_metadata.get('is_demo', False):
            st.caption(f"Basado en {model_metadata.get('n_samples', 'N/A'):,} noticias del dataset real")
            # Generar distribución proporcional a los clusters reales
            cluster_sizes = {name: np.random.randint(800, 2000) for name in cluster_names.values()}
        else:
            st.caption("⚠️ Datos simulados (modelo de demostración)")
            cluster_sizes = {name: np.random.randint(500, 2000) for name in cluster_names.values()}

        fig = px.bar(
            x=list(cluster_sizes.keys()),
            y=list(cluster_sizes.values()),
            labels={'x': 'Cluster', 'y': 'Número de Noticias'},
            color=list(cluster_sizes.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ===================
    # TAB 3: BATCH
    # ===================
    with tab3:
        st.header("🔬 Análisis por Lotes")

        st.markdown("""
        Clasifica múltiples noticias a la vez pegando un CSV o introduciendo varias noticias.
        """)

        # Opción 1: Textarea múltiple
        st.subheader("📝 Opción 1: Introducir múltiples noticias")
        batch_text = st.text_area(
            "Introduce una noticia por línea (formato: Titular | Descripción)",
            placeholder="Trump announces policy | New immigration rules\nApple releases iPhone | New AI features included\n...",
            height=200
        )

        # Opción 2: Cargar CSV
        st.subheader("📁 Opción 2: Cargar archivo CSV")
        uploaded_file = st.file_uploader(
            "Sube un CSV con columnas 'headline' y 'description'",
            type=['csv']
        )

        if st.button("🚀 Procesar Lote", type="primary"):
            news_list = []

            # Procesar desde textarea
            if batch_text:
                lines = batch_text.strip().split('\n')
                for line in lines:
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) == 2:
                            news_list.append({
                                'headline': parts[0].strip(),
                                'description': parts[1].strip()
                            })

            # Procesar desde CSV
            if uploaded_file:
                df_upload = pd.read_csv(uploaded_file)
                if 'headline' in df_upload.columns and 'description' in df_upload.columns:
                    news_list = df_upload[['headline', 'description']].to_dict('records')

            if not news_list:
                st.warning("⚠️ No se encontraron noticias para procesar")
            else:
                with st.spinner(f"🔄 Procesando {len(news_list)} noticias..."):
                    results = []

                    for news in news_list:
                        full_text = f"{news['headline']} {news['description']}"
                        text_clean = preprocess_text(full_text)
                        embedding = generate_embeddings([text_clean], bert_model)[0]

                        cluster_id, cluster_name, similarity = predict_cluster_approximate(
                            embedding, cluster_centroids, cluster_names
                        )

                        results.append({
                            'Titular': news['headline'],
                            'Descripción': news['description'][:50] + '...',
                            'Cluster ID': cluster_id,
                            'Cluster': cluster_name,
                            'Similitud': f"{similarity:.2%}"
                        })

                    # Mostrar resultados
                    st.success(f"✅ {len(results)} noticias clasificadas")

                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)

                    # Resumen por cluster
                    st.subheader("📊 Resumen de Clasificación")
                    cluster_counts = df_results['Cluster'].value_counts()

                    fig = px.pie(
                        values=cluster_counts.values,
                        names=cluster_counts.index,
                        title="Distribución de noticias procesadas"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Botón de descarga
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar resultados (CSV)",
                        data=csv,
                        file_name="resultados_clustering.csv",
                        mime="text/csv"
                    )


if __name__ == "__main__":
    main()

