#!/bin/bash
# =============================================================================
# 🗞️ Semantic News Clusterer - Script Unificado
# =============================================================================
# Este script consolida todas las operaciones del proyecto:
# - Verificación de dependencias
# - Ejecución de Jupyter Notebook
# - Ejecución de Streamlit
# - Flujo completo (entrenar + aplicación)
# - Limpieza de archivos temporales
# =============================================================================

set -e  # Salir si hay error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con colores
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_header() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

# Verificar si estamos en el directorio correcto
check_directory() {
    if [ ! -f "app_streamlit.py" ] || [ ! -f "clustering_noticias_bert_hdbscan.ipynb" ]; then
        print_error "No estás en el directorio del proyecto"
        echo "Por favor, ejecuta este script desde: Semantic_News_Clusterer/"
        exit 1
    fi
}

# Verificar entorno virtual
check_venv() {
    if [ ! -d ".venv" ]; then
        print_error "No se encuentra el entorno virtual .venv"
        echo ""
        echo "Créalo ejecutando:"
        echo "  python3 -m venv .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
}

# Activar entorno virtual
activate_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        print_info "Activando entorno virtual..."
        source .venv/bin/activate
    else
        print_success "Entorno virtual ya activado"
    fi
}

# Verificar dependencias
verify_dependencies() {
    print_info "Verificando dependencias..."

    if python verificar_entorno.py > /dev/null 2>&1; then
        print_success "Todas las dependencias están instaladas"
        return 0
    else
        print_warning "Faltan algunas dependencias"
        echo ""
        read -p "¿Instalar dependencias faltantes? (s/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            pip install -r requirements.txt
            print_success "Dependencias instaladas"
            return 0
        else
            print_error "No se pueden continuar sin las dependencias"
            exit 1
        fi
    fi
}

# Verificar si existe modelo entrenado
check_model() {
    if [ -f "model_data.pkl" ]; then
        local size=$(ls -lh model_data.pkl | awk '{print $5}')
        print_success "Modelo entrenado encontrado (${size})"
        return 0
    else
        print_warning "Modelo no encontrado (se usará demo)"
        return 1
    fi
}

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

# Mostrar ayuda
show_help() {
    cat << EOF
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${GREEN}🗞️  Semantic News Clusterer - Script Unificado${NC}
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${YELLOW}Uso:${NC} ./run.sh [comando]

${YELLOW}Comandos disponibles:${NC}

  ${GREEN}streamlit${NC}         Ejecutar la aplicación web Streamlit
  ${GREEN}notebook${NC}          Abrir Jupyter Notebook
  ${GREEN}full${NC}              Flujo completo (entrenar modelo + streamlit)
  ${GREEN}check${NC}             Verificar instalación y dependencias
  ${GREEN}clean${NC}             Limpiar archivos temporales y cache
  ${GREEN}help${NC}              Mostrar esta ayuda

${YELLOW}Ejemplos:${NC}

  # Ejecutar solo Streamlit (usa modelo demo si no está entrenado)
  ./run.sh streamlit

  # Abrir Jupyter para entrenar modelo
  ./run.sh notebook

  # Flujo completo: entrenar + ejecutar app
  ./run.sh full

  # Verificar que todo está instalado correctamente
  ./run.sh check

${YELLOW}Notas:${NC}

  • El comando 'full' es recomendado para primera vez
  • El modelo se guarda en 'model_data.pkl' al ejecutar el notebook
  • Streamlit usa modelo real si existe, demo si no

${YELLOW}Requisitos:${NC}

  • Python 3.8+
  • Entorno virtual .venv
  • Dependencias instaladas (requirements.txt)

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

EOF
}

# Verificar instalación completa
check_installation() {
    print_header "🔍 VERIFICACIÓN DE INSTALACIÓN"

    # Python
    echo -n "Python versión: "
    python --version || print_error "Python no encontrado"

    # Entorno virtual
    echo ""
    if [ -d ".venv" ]; then
        print_success "Entorno virtual: .venv encontrado"
    else
        print_error "Entorno virtual: No encontrado"
        echo "  Crea uno con: python3 -m venv .venv"
    fi

    # Activar y verificar
    activate_venv

    # Dependencias
    echo ""
    print_info "Verificando dependencias..."
    python verificar_entorno.py

    # Modelo
    echo ""
    if check_model; then
        print_success "Modelo entrenado: Disponible"
    else
        print_info "Modelo entrenado: No disponible (se usará demo)"
    fi

    # Archivos principales
    echo ""
    print_info "Archivos principales:"
    [ -f "app_streamlit.py" ] && echo "  ✓ app_streamlit.py" || echo "  ✗ app_streamlit.py (falta)"
    [ -f "clustering_noticias_bert_hdbscan.ipynb" ] && echo "  ✓ clustering_noticias_bert_hdbscan.ipynb" || echo "  ✗ clustering_noticias_bert_hdbscan.ipynb (falta)"
    [ -f "requirements.txt" ] && echo "  ✓ requirements.txt" || echo "  ✗ requirements.txt (falta)"

    echo ""
    print_success "Verificación completada"
}

# Ejecutar Streamlit
run_streamlit() {
    print_header "🌐 EJECUTANDO APLICACIÓN STREAMLIT"

    check_directory
    check_venv
    activate_venv
    verify_dependencies

    echo ""
    if check_model; then
        print_success "Usando modelo REAL entrenado"
    else
        print_info "Usando modelo de DEMOSTRACIÓN"
        print_warning "Para usar modelo real, ejecuta primero: ./run.sh notebook"
    fi

    echo ""
    print_info "Iniciando Streamlit..."
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  La app se abrirá en: ${GREEN}http://localhost:8501${NC}"
    echo "  Para detener: ${YELLOW}Ctrl+C${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    streamlit run app_streamlit.py
}

# Ejecutar Jupyter Notebook
run_notebook() {
    print_header "📓 ABRIENDO JUPYTER NOTEBOOK"

    check_directory
    check_venv
    activate_venv
    verify_dependencies

    echo ""
    print_info "Abriendo Jupyter Notebook..."
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📝 INSTRUCCIONES:"
    echo "  1. Abre: ${GREEN}clustering_noticias_bert_hdbscan.ipynb${NC}"
    echo "  2. Ejecuta: ${YELLOW}Cell → Run All${NC}"
    echo "  3. Espera ~15 minutos (con 10,000 noticias)"
    echo "  4. Verifica mensaje: ${GREEN}'✅ MODELO LISTO'${NC}"
    echo "  5. El modelo se guardará en: ${GREEN}model_data.pkl${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    read -p "Presiona Enter para continuar..."

    jupyter notebook
}

# Flujo completo
run_full() {
    print_header "🚀 FLUJO COMPLETO: ENTRENAR MODELO + STREAMLIT"

    check_directory
    check_venv
    activate_venv
    verify_dependencies

    echo ""
    print_info "Este flujo incluye:"
    echo "  1. Verificar si existe modelo entrenado"
    echo "  2. Opción de entrenar nuevo modelo (Jupyter)"
    echo "  3. Ejecutar aplicación Streamlit con modelo"
    echo ""

    # Verificar modelo existente
    if check_model; then
        echo ""
        read -p "¿Re-entrenar modelo? (s/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            TRAIN=true
        else
            TRAIN=false
            print_info "Usando modelo existente"
        fi
    else
        echo ""
        print_warning "Modelo no encontrado"
        echo ""
        echo "${YELLOW}Opciones:${NC}"
        echo "  ${GREEN}1.${NC} Abrir Jupyter Notebook para entrenar modelo"
        echo "  ${GREEN}2.${NC} Usar modelo de demostración en Streamlit"
        echo ""
        read -p "¿Qué quieres hacer? (1/2): " -n 1 -r
        echo

        if [[ $REPLY == "1" ]]; then
            echo ""
            print_info "Abriendo Jupyter Notebook..."
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  📝 INSTRUCCIONES:"
            echo "  1. Abre: clustering_noticias_bert_hdbscan.ipynb"
            echo "  2. Ejecuta: Cell → Run All"
            echo "  3. Espera ~15 minutos"
            echo "  4. Verifica: '✅ MODELO LISTO PARA USAR'"
            echo "  5. Cierra el navegador y vuelve aquí"
            echo "  6. Ejecuta de nuevo: ./run.sh streamlit"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            read -p "Presiona Enter para continuar..."
            jupyter notebook
            exit 0
        else
            print_info "Continuando con modelo de demostración..."
            TRAIN=false
        fi
    fi

    # Ejecutar Streamlit
    echo ""
    print_header "🌐 EJECUTANDO STREAMLIT"

    if [ "$TRAIN" = false ]; then
        if [ -f "model_data.pkl" ]; then
            print_success "Usando modelo real: model_data.pkl"
        else
            print_warning "Usando modelo de demostración"
        fi
    fi

    echo ""
    print_info "Iniciando aplicación..."
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  La app se abrirá en: http://localhost:8501"
    echo "  Para detener: Ctrl+C"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    streamlit run app_streamlit.py
}

# Limpiar archivos temporales
clean_files() {
    print_header "🧹 LIMPIANDO ARCHIVOS TEMPORALES"

    check_directory

    echo ""
    print_info "Archivos a eliminar:"

    # Listar archivos
    [ -d "__pycache__" ] && echo "  • __pycache__/"
    [ -d ".ipynb_checkpoints" ] && echo "  • .ipynb_checkpoints/"
    [ -f ".DS_Store" ] && echo "  • .DS_Store"
    find . -name "*.pyc" -type f 2>/dev/null && echo "  • Archivos .pyc"

    echo ""
    read -p "¿Confirmar limpieza? (s/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Ss]$ ]]; then
        # Eliminar archivos
        rm -rf __pycache__ 2>/dev/null
        rm -rf .ipynb_checkpoints 2>/dev/null
        rm -f .DS_Store 2>/dev/null
        find . -name "*.pyc" -type f -delete 2>/dev/null

        print_success "Archivos temporales eliminados"

        # Opcional: limpiar cache de Streamlit
        echo ""
        read -p "¿Limpiar también cache de Streamlit? (s/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            streamlit cache clear
            print_success "Cache de Streamlit limpiado"
        fi
    else
        print_info "Limpieza cancelada"
    fi
}

# =============================================================================
# MAIN
# =============================================================================

# Si no hay argumentos, mostrar ayuda
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Procesar comando
case "$1" in
    streamlit)
        run_streamlit
        ;;
    notebook)
        run_notebook
        ;;
    full)
        run_full
        ;;
    check)
        check_installation
        ;;
    clean)
        clean_files
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Comando desconocido: $1"
        echo ""
        echo "Usa: ./run.sh help"
        echo ""
        exit 1
        ;;
esac

exit 0

