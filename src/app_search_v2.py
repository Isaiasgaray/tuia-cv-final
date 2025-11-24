import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import faiss
import gradio as gr
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torchvision import models, transforms

OUTPUT_DIR = 'data/processed'
INDEX_R18_FILENAME = 'faiss_index_R18.bin'
INDEX_R50_FILENAME = 'faiss_index_R50.bin'
WEIGHTS_R18_PATH = 'data/resnet18_70_breeds_best_weights.pth'
PATHS_FILENAME = 'image_paths.npy'
K_SIMILAR = 10

FAISS_PATH_R18 = os.path.join(OUTPUT_DIR, INDEX_R18_FILENAME)
FAISS_PATH_R50 = os.path.join(OUTPUT_DIR, INDEX_R50_FILENAME)
PATHS_PATH = os.path.join(OUTPUT_DIR, PATHS_FILENAME)

MODELS = {} 
FAISS_INDEXES = {}
IMAGE_PATHS = None
SYSTEM_READY = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_image_keras(img_input, target_size=(224, 224)):
    """Pre-procesa una imagen para el modelo Keras (ResNet50)."""
    img = img_input.resize(target_size)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def load_pytorch_resnet18(num_classes=70):
    """Carga el ResNet18 de PyTorch con los pesos entrenados para feature extraction."""
    if not os.path.exists(WEIGHTS_R18_PATH):
        raise FileNotFoundError(f"CRÍTICO: Faltan pesos de PyTorch: {WEIGHTS_R18_PATH}")
        
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Cargar los pesos entrenados
    model.load_state_dict(torch.load(WEIGHTS_R18_PATH, map_location=DEVICE))
    
    model.fc = nn.Identity() 
    model = model.to(DEVICE)
    model.eval()
    return model

def get_pytorch_transforms():
    """Define las transformaciones estándar para el modelo PyTorch."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_breed_from_path(path):
    """Extrae el nombre de la raza."""
    return os.path.basename(os.path.dirname(path)).replace('_', ' ')

def initialize_search_system():
    """Carga ambos modelos y sus respectivos índices FAISS, con debug explícito."""
    global MODELS, FAISS_INDEXES, IMAGE_PATHS, SYSTEM_READY
    
    print("Iniciando el sistema de búsqueda multi-modelo...")
    SYSTEM_READY = False
    
    # Cargar las rutas de imagen primero
    if not os.path.exists(PATHS_PATH):
        print(f"FALLO CRÍTICO: No se encontraron las rutas de imagen en {PATHS_PATH}. Ejecuta el script de generación de embeddings.")
        return
    IMAGE_PATHS = np.load(PATHS_PATH)
    
    # --- ResNet50 Keras/TF ---
    try:
        print("Intentando cargar ResNet50 de Keras...")
        if not os.path.exists(FAISS_PATH_R50):
            raise FileNotFoundError(f"CRÍTICO: Faltan índices FAISS R50: {FAISS_PATH_R50}")
            
        MODELS['ResNet50'] = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        FAISS_INDEXES['ResNet50'] = faiss.read_index(FAISS_PATH_R50)
        print("✅ Modelo ResNet50 cargado exitosamente.")
        
    except Exception as e:
        print(f"Falló la carga del Modelo ResNet50. Detalle: {e}")
        
    # --- ResNet18 PyTorch ---
    try:
        print("Intentando cargar Modelo ResNet18 PyTorch...")
        if not os.path.exists(FAISS_PATH_R18):
            raise FileNotFoundError(f"CRÍTICO: Faltan índices FAISS R18: {FAISS_PATH_R18}")
            
        MODELS['ResNet18'] = load_pytorch_resnet18()
        FAISS_INDEXES['ResNet18'] = faiss.read_index(FAISS_PATH_R18)
        print("✅ Modelo ResNet18 cargado exitosamente.")
        
    except Exception as e:
        print(f"Falló la carga del ResNet18. Detalle: {e}")

    # Chequeo final
    if not MODELS:
        print("Error: No se pudo cargar ningún modelo.")
        return

    print(f"Sistema inicializado. {len(MODELS)} modelos disponibles: {list(MODELS.keys())}.")
    SYSTEM_READY = True

# Llama a la inicialización
initialize_search_system()

def get_embedding(model_key, input_img_pil):
    """Extrae el vector de features usando el modelo Keras o PyTorch apropiado."""
    model = MODELS[model_key]
    
    if model_key == 'ResNet50':
        # Keras/TensorFlow Lógica
        processed_img = load_and_preprocess_image_keras(input_img_pil)
        embedding = model.predict(processed_img, verbose=0).flatten().astype('float32')
        
    elif model_key == 'ResNet18':
        # PyTorch Lógica
        preprocess = get_pytorch_transforms()
        input_tensor = preprocess(input_img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convertir Tensor de PyTorch a array numpy float32
        embedding = output.squeeze().cpu().numpy().astype('float32')
        
    else:
        raise ValueError(f"Modelo {model_key} no soportado.")

    return np.expand_dims(embedding, axis=0)

def find_similar_images(input_img_pil, model_key_selection):
    """
    Procesa la imagen de entrada, busca las K_SIMILAR imágenes más parecidas
    usando el modelo seleccionado y devuelve los resultados.
    """
    if not SYSTEM_READY or model_key_selection not in MODELS:
        msg = f"Error: Modelo '{model_key_selection}' no disponible o sistema no inicializado."
        return None, [(None, msg)], msg

    # Seleccionar el índice FAISS correcto
    faiss_index = FAISS_INDEXES[model_key_selection]
    
    # Extracción del embedding de la consulta
    query_embedding = get_embedding(model_key_selection, input_img_pil)

    # Búsqueda con FAISS
    # D: Distancias, I: Índices (identificadores de los vectores)
    D, I = faiss_index.search(query_embedding, K_SIMILAR + 1)

    # Procesamiento de resultados
    start_index = 0
    if D[0][0] < 1e-6:
        start_index = 1
    
    result_indices = I[0][start_index:K_SIMILAR + start_index]
    similar_image_paths = IMAGE_PATHS[result_indices]

    # Clasificación por voto mayoritario
    retrieved_breeds = [get_breed_from_path(path) for path in similar_image_paths]
    breed_counts = Counter(retrieved_breeds)
    predicted_breed = breed_counts.most_common(1)[0][0]

    results_list = []
    
    # Calcular métricas para la galería
    for i, path in enumerate(similar_image_paths):
        distance = D[0][i + start_index] 
        min_dist = D[0][start_index]
        score = min_dist / distance 
        
        img = Image.open(path).resize((224, 224))
        
        results_list.append((img, f"Similitud: {score:.2f} | Raza: {get_breed_from_path(path)}"))

    return input_img_pil.resize((224, 224)), results_list, f"Raza predicha: {predicted_breed} | Modelo: {model_key_selection}"

if SYSTEM_READY:
    
    # Componente para mostrar las imágenes encontradas
    gallery_output = gr.Gallery(
        label="10 Imágenes Más Similares Encontradas", 
        columns=5, 
        rows=2, 
        object_fit="contain",
        height=450
    )
    
    model_options = list(MODELS.keys())
    
    with gr.Blocks(title="Búsqueda de Razas de Perro por Similitud Multi-Modelo") as app:
        gr.Markdown(
            "# 🐕 Búsqueda de Imágenes Similares (Image Retrieval)\n"
            "**Etapa 2, punto 2:** Seleccione el modelo para generar los vectores de características (Embeddings)."
        )

        predicted_output = gr.Textbox(label="Clasificación por Voto Mayoritario (K=10)", value="Esperando consulta...", scale=1)

        with gr.Row():
            # Entrada
            input_image = gr.Image(type="pil", label="Imagen de Entrada (Consulta)", width=300)
            
            model_selector = gr.Dropdown(
                model_options,
                label="Seleccione el Modelo de Características",
                value=model_options[0],
                scale=1
            )
            
            # Botón de búsqueda
            search_button = gr.Button("Buscar Imágenes Similares", scale=1)
        
        with gr.Row():
            # Salida
            input_display = gr.Image(type="pil", label="Imagen de Entrada (Procesada)", width=300)
            
            # Galería de resultados
            gallery_output.render()

        # Conectar el botón con la función
        search_button.click(
            fn=find_similar_images,
            inputs=[input_image, model_selector],
            outputs=[input_display, gallery_output, predicted_output]
        )
        
    # Lanzar la aplicación
    app.launch()
else:
    print("No se pudo lanzar la aplicación.")