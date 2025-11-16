import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import faiss
import glob

DATASET_PATH = 'dataset'
OUTPUT_DIR = 'data/processed'
INDEX_FILENAME = 'faiss_index.bin'
PATHS_FILENAME = 'image_paths.npy'
EMBEDDING_DIM = 2048

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Carga y pre-procesa una imagen para el modelo"""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expanded)
    except Exception as e:
        print(f"Error procesando la imagen {img_path}: {e}")
        return None

def extract_embeddings(base_model, image_paths):
    """
    Utiliza un modelo pre-entrenado para 
    extraer un embedding de cada imagen.
    """
    print(f"Comenzando la extracción de {len(image_paths)} embeddings...")
    embeddings = []
    processed_paths = []
    
    for i, path in enumerate(image_paths):
        if (i + 1) % 500 == 0:
            print(f" > Procesando imagen {i + 1}/{len(image_paths)}")
            
        processed_img = load_and_preprocess_image(path)
        
        if processed_img:
            embedding = base_model.predict(processed_img, verbose=0) 
            embeddings.append(embedding.flatten())
            processed_paths.append(path)
            
    return np.array(embeddings, dtype='float32'), processed_paths

def create_vector_db():
    """
    Función principal para crear la base de datos vectorial.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Cargando modelo...")
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    print(f"Buscando imágenes en: {DATASET_PATH}")
    all_paths = glob.glob(os.path.join(DATASET_PATH, '**', '*.jpg'), recursive=True)
    if not all_paths:
        print(f"ERROR: No se encontraron imágenes en {DATASET_PATH}. Verifica la ruta.")
        return

    # Extracción de embeddings
    embeddings, final_paths = extract_embeddings(base_model, all_paths)
    print(f"Extracción finalizada. Se generaron {embeddings.shape[0]} embeddings de dimensión {embeddings.shape[1]}.")

    # Indexar vectores 
    print("Indexando embeddings...")
    dimension = embeddings.shape[1]
    # Usamos un índice simple de L2 (distancia euclidiana)
    index = faiss.IndexFlatL2(dimension)
    # Agregar los vectores al índice
    index.add(embeddings)
    print(f"Base de datos creada. Total de vectores indexados: {index.ntotal}")

    # Guardar los artefactos
    faiss_path = os.path.join(OUTPUT_DIR, INDEX_FILENAME)
    paths_path = os.path.join(OUTPUT_DIR, PATHS_FILENAME)
    
    # Guardar el índice y las rutas correspondientes
    faiss.write_index(index, faiss_path)
    np.save(paths_path, np.array(final_paths))
    
    print("\n--- ¡Proceso Completado! ---")
    print(f"✅ Índice guardado en: {faiss_path}")
    print(f"✅ Rutas de imagen guardadas en: {paths_path}")
    
if __name__ == "__main__":
    create_vector_db()