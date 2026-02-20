"""
Script de verificación del entorno de Semantic News Clusterer
Verifica que todas las dependencias estén instaladas correctamente.
"""

import sys

def verificar_instalacion():
    """Verifica que todas las librerías necesarias estén instaladas."""

    print("=" * 80)
    print("🔍 VERIFICACIÓN DEL ENTORNO - Semantic News Clusterer")
    print("=" * 80)
    print()

    librerias = [
        ('pandas', 'Manejo de datos'),
        ('numpy', 'Operaciones numéricas'),
        ('jupyter', 'Notebook interactivo'),
        ('sentence_transformers', 'Embeddings BERT'),
        ('umap', 'Reducción dimensional'),
        ('hdbscan', 'Clustering'),
        ('plotly', 'Visualización interactiva'),
        ('seaborn', 'Visualización estática'),
        ('matplotlib', 'Gráficos'),
        ('sklearn', 'Machine Learning'),
        ('kagglehub', 'Descarga de datos'),
        ('tqdm', 'Barras de progreso'),
        ('torch', 'Deep Learning backend'),
        ('transformers', 'Modelos NLP')
    ]

    print("📦 Verificando librerías instaladas:\n")

    errores = []
    exitosas = []

    for nombre, descripcion in librerias:
        try:
            __import__(nombre)
            exitosas.append((nombre, descripcion))
            print(f"✅ {nombre:25} - {descripcion}")
        except ImportError as e:
            errores.append((nombre, descripcion, str(e)))
            print(f"❌ {nombre:25} - {descripcion} [ERROR]")

    print()
    print("=" * 80)
    print(f"📊 RESUMEN: {len(exitosas)}/{len(librerias)} librerías instaladas correctamente")
    print("=" * 80)

    if errores:
        print("\n⚠️  ERRORES ENCONTRADOS:\n")
        for nombre, descripcion, error in errores:
            print(f"   ❌ {nombre}: {descripcion}")
            print(f"      Error: {error}\n")
        print("💡 Solución: Ejecuta 'pip install -r requirements.txt' en el entorno virtual\n")
        return False
    else:
        print("\n✅ Todas las dependencias están instaladas correctamente")
        print("\n🚀 Puedes ejecutar el notebook con: jupyter notebook\n")
        return True


def verificar_python():
    """Verifica la versión de Python."""
    version = sys.version_info
    print(f"\n🐍 Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("✅ Versión de Python compatible")
        return True
    else:
        print("⚠️  Se recomienda Python 3.8 o superior")
        return False


if __name__ == "__main__":
    print("\n")
    verificar_python()
    print()
    exito = verificar_instalacion()

    if exito:
        print("=" * 80)
        print("🎉 ¡El entorno está listo para usar!")
        print("=" * 80)
        print("\n📝 Próximos pasos:")
        print("   1. Activar el entorno virtual: source .venv/bin/activate")
        print("   2. Lanzar Jupyter: jupyter notebook")
        print("   3. Abrir: semantic_news_clustering.ipynb")
        print()
        sys.exit(0)
    else:
        print("=" * 80)
        print("⚠️  Por favor, instala las dependencias faltantes")
        print("=" * 80)
        sys.exit(1)

