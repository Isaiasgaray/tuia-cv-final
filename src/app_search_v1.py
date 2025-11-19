import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import faiss
import gradio as gr
from PIL import Image
from collections import Counter

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
OUTPUT_DIR = 'data/processed'
INDEX_FILENAME = 'faiss_index.bin'
PATHS_FILENAME = 'image_paths.npy'
# Número de imágenes más similares a buscar
K_SIMILAR = 10

# Rutas completas
FAISS_PATH = os.path.join(OUTPUT_DIR, INDEX_FILENAME)
PATHS_PATH = os.path.join(OUTPUT_DIR, PATHS_FILENAME)

# Variables globales para el modelo y la base de datos
BASE_MODEL  = None
FAISS_INDEX = None
IMAGE_PATHS = None

def load_and_preprocess_image(img_input, target_size=(224, 224)):
    """Pre-procesa una imagen para el modelo."""
    if isinstance(img_input, str):
        # Si es una ruta (útil para la extracción inicial, no usado directamente en Gradio)
        img = image.load_img(img_input, target_size=target_size)
    else:
        # Si es una imagen PIL (recibida de Gradio)
        img = img_input.resize(target_size)

    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def initialize_search_system():
    """
    Carga el modelo ResNet50 y el índice FAISS una sola vez.
    Esto acelera la aplicación Gradio.
    """
    global BASE_MODEL, FAISS_INDEX, IMAGE_PATHS
    
    print("Iniciando el sistema de búsqueda...")

    print("Cargando modelo ResNet50...")
    BASE_MODEL = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Cargar el índice FAISS y las rutas
    if not os.path.exists(FAISS_PATH) or not os.path.exists(PATHS_PATH):
        raise FileNotFoundError(
            f"ERROR: No se encontraron los archivos de la Base de Datos Vectorial en {OUTPUT_DIR}. "
            "Asegúrate de ejecutar 'create_vector_db.py' primero."
        )
    
    print(f"Cargando índice FAISS desde: {FAISS_PATH}")
    FAISS_INDEX = faiss.read_index(FAISS_PATH)
    
    print(f"Cargando rutas de imagen desde: {PATHS_PATH}")
    IMAGE_PATHS = np.load(PATHS_PATH)
    
    print(f"Sistema inicializado. {FAISS_INDEX.ntotal} vectores cargados.")
    return True

# Llama a la inicialización al ejecutar el script
try:
    SYSTEM_READY = initialize_search_system()
except Exception as e:
    print(f"FALLO CRÍTICO AL INICIALIZAR: {e}")
    SYSTEM_READY = False


# --- FUNCIÓN PRINCIPAL DE BÚSQUEDA ---

def find_similar_images(input_img_pil):
    """
    Procesa la imagen de entrada, busca las K_SIMILAR imágenes más parecidas
    en la Base de Datos Vectorial y las devuelve.
    """
    if not SYSTEM_READY:
        return None, ["Error: Sistema no inicializado. Ver logs."]

    # Extracción del embedding de la consulta
    processed_img = load_and_preprocess_image(input_img_pil)
    
    # El embedding de la consulta debe ser float32 para FAISS
    query_embedding = BASE_MODEL.predict(processed_img, verbose=0).flatten().astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # Búsqueda con FAISS
    # D: Distancias, I: Índices (identificadores de los vectores)
    # Buscamos K_SIMILAR + 1 para tener en cuenta la auto-imagen si está en el dataset
    D, I = FAISS_INDEX.search(query_embedding, K_SIMILAR + 1)

    # Procesamiento de resultados
    
    # Se omiten el primer resultado si la distancia es cero (misma imagen)
    start_index = 0
    # Chequea si la distancia al primer resultado es cercana a cero
    if D[0][0] < 1e-6:
        start_index = 1
    
    # Obtener los K_SIMILAR mejores resultados (excluyendo el primero si es la misma imagen)
    result_indices = I[0][start_index:K_SIMILAR + start_index]
    
    # Obtener las rutas correspondientes
    similar_image_paths = IMAGE_PATHS[result_indices]

    # --- LÓGICA DE CLASIFICACIÓN POR VOTO MAYORITARIO (K-NN) ---
    retrieved_breeds = [get_breed_from_path(path) for path in similar_image_paths]
    breed_counts = Counter(retrieved_breeds)
    predicted_breed = breed_counts.most_common(1)[0][0]

    results_list = []
    for i, path in enumerate(similar_image_paths):
        # D[0] contiene las distancias para la consulta
        distance = D[0][i + start_index] 
        
        # Obtenemos la distancia mínima válida (la que corresponde al índice start_index)
        min_dist = D[0][start_index]
        
        # Calculamos un score. Usar min_dist / distance asegura que el score sea 1.00 para la mejor coincidencia.
        # El valor score representa: (Qué tan cerca está la imagen de la mejor coincidencia)
        score = min_dist / distance 
        
        img = Image.open(path).resize((224, 224))
        
        results_list.append((img, f"Similitud: {score:.2f} | path: {path}"))

    return input_img_pil.resize((224, 224)), results_list, f"Raza predicha: {predicted_breed}"


# --- FUNCIÓN AUXILIAR PARA OBTENER LA RAZA ---

def get_breed_from_path(path):
    """
    Extrae el nombre de la raza asumiendo que es el nombre
    de la carpeta inmediatamente anterior al archivo.

        Ejemplo: dataset/test/Golden Retriever/01.jpg -> Golden Retriever
    """
    return os.path.basename(os.path.dirname(path)).replace('_', ' ')

# --- INTERFAZ GRADIO ---

if SYSTEM_READY:
    
    # Componente para mostrar las imágenes encontradas
    gallery_output = gr.Gallery(
        label="10 Imágenes Más Similares Encontradas", 
        columns=5, 
        rows=2, 
        object_fit="contain",
        height=450
    )

    with gr.Blocks(title="Búsqueda de Razas de Perro por Similitud") as app:
        gr.Markdown(
            "# 🐕 Búsqueda de Imágenes Similares (Image Retrieval)\n"
            "Sube una imagen de un perro. El sistema usará ResNet50 y FAISS para encontrar las 10 imágenes más parecidas en el dataset."
        )

        predicted_output = gr.Textbox(label="Clasificación por Voto Mayoritario (K=10)", value="Esperando consulta...", scale=1)

        with gr.Row():
            # Entrada
            input_image = gr.Image(type="pil", label="Imagen de Entrada (Consulta)", width=300)
            
            # Botón de búsqueda
            search_button = gr.Button("Buscar Imágenes Similares")
        
        with gr.Row():
            # Salida
            input_display = gr.Image(type="pil", label="Imagen de Entrada (Procesada)", width=300)
            
            # Galería de resultados
            gallery_output.render()


        # Conectar el botón con la función
        search_button.click(
            fn=find_similar_images,
            inputs=[input_image],
            outputs=[input_display, gallery_output, predicted_output]
        )
        
    # Lanzar la aplicación
    app.launch()
else:
    print("\nNo se pudo lanzar la aplicación Gradio debido a errores de inicialización. Revisa los mensajes de 'FileNotFoundError'.")
