import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import faiss
import glob
from collections import Counter

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'data/processed'
INDEX_FILENAME = 'faiss_index.bin'
PATHS_FILENAME = 'image_paths.npy'

FAISS_PATH = os.path.join(OUTPUT_DIR, INDEX_FILENAME)
PATHS_PATH = os.path.join(OUTPUT_DIR, PATHS_FILENAME)

# Parámetros de prueba
# TEST_DATASET_PATH = 'dataset_test' # Carpeta con el conjunto de prueba
TEST_DATASET_PATH = 'dataset/test' # Carpeta con el conjunto de prueba
K_NDCG = 10 # Se evalúa la calidad del ranking hasta el puesto 10

# --- FUNCIONES AUXILIARES ---


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


def get_breed_from_path(path):
    """Extrae la raza de perro de la ruta."""
    # Asume que la raza es el nombre de la carpeta
    return os.path.basename(os.path.dirname(path))

def load_system():
    """Carga el modelo y el índice FAISS."""
    # La función load_and_preprocess_image se omite aquí por brevedad, asume que está definida.
    # Usaremos una versión simplificada de la carga de la interfaz.

    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    if not os.path.exists(FAISS_PATH) or not os.path.exists(PATHS_PATH):
        raise FileNotFoundError(
            f"ERROR: No se encontraron los archivos necesarios en {OUTPUT_DIR}."
        )
        
    faiss_index = faiss.read_index(FAISS_PATH)
    image_paths = np.load(PATHS_PATH)
    
    return base_model, faiss_index, image_paths

def calculate_ndcg(relevance_scores, k):
    """Calcula la métrica NDCG@k."""
    # 1. Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i in range(k):
        # Ganancia (G) = relevancia, Descuento = log2(i + 2)
        if i < len(relevance_scores):
            dcg += relevance_scores[i] / np.log2(i + 2)
    
    # 2. Ideal Discounted Cumulative Gain (IDCG)
    # Para la NDCG, el IDCG se calcula ordenando las relevancias idealmente (descendente)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i in range(k):
        if i < len(ideal_scores):
            idcg += ideal_scores[i] / np.log2(i + 2)
            
    # 3. Normalization (NDCG)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_system():
    """Función principal para evaluar el sistema con NDCG@10."""
    try:
        base_model, faiss_index, image_paths = load_system()
    except Exception as e:
        print(f"Error cargando sistema: {e}")
        return

    # Obtener todas las imágenes de prueba
    test_paths = glob.glob(os.path.join(TEST_DATASET_PATH, '**', '*.jpg'), recursive=True)
    if not test_paths:
        print(f"ERROR: No se encontraron imágenes en {TEST_DATASET_PATH}. Verifica la ruta.")
        return

    all_ndcgs = []

    print(f"Comenzando la evaluación con {len(test_paths)} imágenes de prueba...")
    
    for i, test_path in enumerate(test_paths):
        # 1. Preparar la imagen de consulta y extraer el embedding
        try:
            processed_img = load_and_preprocess_image(test_path)
            query_embedding = base_model.predict(processed_img, verbose=0).flatten().astype('float32')
            query_embedding = np.expand_dims(query_embedding, axis=0)
        except Exception as e:
            print(f"Saltando imagen de prueba {test_path}: {e}")
            continue

        # 2. Buscar K resultados
        # Buscamos K_NDCG + 1 para omitir la auto-imagen
        D, I = faiss_index.search(query_embedding, K_NDCG + 1)
        
        # 3. Determinar la verdad fundamental (Ground Truth)
        true_breed = get_breed_from_path(test_path)
        
        # 4. Procesar Resultados y Calcular Relevancia
        start_index = 0
        if D[0][0] < 1e-6:
            start_index = 1
            
        retrieved_indices = I[0][start_index:K_NDCG + start_index]
        
        # Si no hay suficientes resultados, rellenar con índices inválidos
        if len(retrieved_indices) < K_NDCG:
            retrieved_indices = np.pad(retrieved_indices, (0, K_NDCG - len(retrieved_indices)), constant_values=-1)

        relevance_scores = []
        for rank in range(K_NDCG):
            idx = retrieved_indices[rank]
            if idx == -1:
                # No hay resultado, relevancia 0
                relevance_scores.append(0)
                continue
                
            retrieved_path = image_paths[idx]
            retrieved_breed = get_breed_from_path(retrieved_path)
            
            # Relevancia binaria: 1 si la raza coincide, 0 si no.
            relevance = 1 if retrieved_breed == true_breed else 0
            relevance_scores.append(relevance)
        
        # 5. Calcular NDCG@10 para esta consulta
        ndcg_k = calculate_ndcg(relevance_scores, K_NDCG)
        all_ndcgs.append(ndcg_k)
        
        # print(f" {i+1}/{len(test_paths)}: {true_breed} -> NDCG@{K_NDCG}: {ndcg_k:.4f}")

    # 6. Resultado Final
    if all_ndcgs:
        mean_ndcg = np.mean(all_ndcgs)
        print("\n--- RESULTADOS DE LA EVALUACIÓN ---")
        print(f"Métrica: NDCG@{K_NDCG}")
        print(f"Número total de consultas de prueba: {len(all_ndcgs)}")
        print(f"NDCG Promedio del Sistema: {mean_ndcg:.4f}")
    else:
        print("No se pudo calcular el NDCG promedio. Verifica los datos de prueba.")
        
if __name__ == "__main__":
    # Necesitas la definición de load_and_preprocess_image del otro archivo aquí o importarla.
    # Por simplicidad, se puede copiar/pegar la función load_and_preprocess_image aquí.
    
    # ... (Copiar la función load_and_preprocess_image del app_search.py aquí) ...
    
    evaluate_system()