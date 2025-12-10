import numpy as np
import os
import faiss
import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import torch
import torch.nn as nn
from torchvision import models, transforms

# --- CONFIGURACIÓN GLOBAL ---
DATASET_PATH = 'dataset'
OUTPUT_DIR = 'data/processed'
WEIGHTS_R18_PATH = 'data/resnet18_70_breeds_best_weights.pth' 
PATHS_FILENAME = 'image_paths.npy'
# Número de razas
NUM_CLASSES = 70
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- FUNCIONES DE SOPORTE GENERAL ---

def get_all_image_paths():
    """Busca todas las imágenes en las carpetas train"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Buscando imágenes en: {DATASET_PATH}")
    # Busca imágenes en todas las subcarpetas del dataset
    all_paths = glob.glob(os.path.join(DATASET_PATH, 'train/**', '*.jpg'), recursive=True)
    if not all_paths:
        raise FileNotFoundError(f"ERROR: No se encontraron imágenes en {DATASET_PATH}. Verifica la ruta.")
    return all_paths

def save_faiss_index(embeddings, final_paths, index_filename):
    """Indexa y guarda la base de datos FAISS y las rutas."""
    faiss_path = os.path.join(OUTPUT_DIR, index_filename)
    paths_path = os.path.join(OUTPUT_DIR, PATHS_FILENAME)

    dimension = embeddings.shape[1]
    print(f"Indexando {embeddings.shape[0]} embeddings de dimensión {dimension}...")
    
    # Usamos un índice simple de L2 (distancia euclidiana)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, faiss_path)
    np.save(paths_path, np.array(final_paths)) 
    
    print(f"Índice ({index_filename}) guardado en: {faiss_path}")
    print(f"Rutas de imagen guardadas en: {paths_path}")

# --- ResNet50 ---

def load_and_preprocess_image_keras(img_path, target_size=(224, 224)):
    """Carga y pre-procesa una imagen para el modelo de Keras."""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expanded)
    except Exception as e:
        print(f"Error procesando la imagen {img_path}: {e}")
        return None

def extract_embeddings_keras(base_model, image_paths):
    """Extrae embeddings usando Keras/TensorFlow."""
    embeddings = []
    processed_paths = []
    
    for i, path in enumerate(image_paths):
        if (i + 1) % 500 == 0:
            print(f" > Keras - Procesando imagen {i + 1}/{len(image_paths)}")

        processed_img = load_and_preprocess_image_keras(path)
        
        if processed_img is not None:
            # La salida del pooling es el embedding
            embedding = base_model.predict(processed_img, verbose=0) 
            embeddings.append(embedding.flatten())
            processed_paths.append(path)
            
    return np.array(embeddings, dtype='float32'), processed_paths

def create_db_resnet50(image_paths):
    """Genera el índice FAISS para ResNet50."""
    print("--- INICIANDO GENERACIÓN: ResNet50 ---")
    
    # Cargar modelo (weights='imagenet' por defecto para el Modelo)
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Extracción de embeddings
    embeddings, final_paths = extract_embeddings_keras(base_model, image_paths)
    print(f"Extracción Keras finalizada. Generados {embeddings.shape[0]} embeddings.")

    # Guardar el índice
    save_faiss_index(embeddings, final_paths, 'faiss_index_R50.bin')

# --- ResNet18 ---

def get_pytorch_transforms():
    """Define las transformaciones estándar de PyTorch."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_pytorch_resnet18():
    """Carga ResNet18, los pesos afinados y lo adapta para feature extraction."""
    
    # Cargar la arquitectura y pesos
    if not os.path.exists(WEIGHTS_R18_PATH):
        raise FileNotFoundError(f"Error: No se encontraron los pesos afinados de PyTorch en {WEIGHTS_R18_PATH}.")
        
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # Recrear la capa para cargar el state_dict
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_R18_PATH, map_location=DEVICE))
    
    # Reemplazar la capa final por Identidad para obtener el embedding
    model.fc = nn.Identity() 
    model = model.to(DEVICE)
    model.eval() 
    return model

def extract_embeddings_pytorch(base_model, image_paths):
    """Extrae embeddings usando PyTorch."""
    embeddings = []
    processed_paths = []
    data_transform = get_pytorch_transforms()

    with torch.no_grad(): 
        for i, path in enumerate(image_paths):
            if (i + 1) % 500 == 0:
                print(f" > PyTorch - Procesando imagen {i + 1}/{len(image_paths)}")

            try:
                img = Image.open(path).convert('RGB')
                input_tensor = data_transform(img).unsqueeze(0).to(DEVICE)
                output = base_model(input_tensor)
                
                # Convertir Tensor de PyTorch (512 dim) a array numpy float32
                embedding = output.squeeze().cpu().numpy().astype('float32')
                
                embeddings.append(embedding)
                processed_paths.append(path)
            
            except Exception as e:
                print(f"Error procesando la imagen {path}: {e}")
                continue
            
    return np.array(embeddings, dtype='float32'), processed_paths

def create_db_resnet18(image_paths):
    """Genera el índice FAISS para ResNet18."""
    print("\n--- INICIANDO GENERACIÓN: ResNet18 ---")
    
    try:
        # Cargar el modelo ResNet18 con pesos
        base_model = load_pytorch_resnet18()
    except FileNotFoundError as e:
        print(f"ERROR: No se pudo generar el índice R18. {e}")
        return

    # Extracción de embeddings
    embeddings, final_paths = extract_embeddings_pytorch(base_model, image_paths)
    print(f"Extracción PyTorch finalizada. Generados {embeddings.shape[0]} embeddings.")
    
    # Guardar el índice
    save_faiss_index(embeddings, final_paths, 'faiss_index_R18.bin')

# --- main ---

all_paths = get_all_image_paths()

create_db_resnet50(all_paths)

create_db_resnet18(all_paths)

print("Generación de bases completada.")