from ultralytics import YOLO
import os
from PIL import ImageDraw
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import faiss
import gradio as gr
from PIL import Image
from collections import Counter

# --- Detector YOLO ---
try:
    YOLO_MODEL = YOLO('yolov8n.pt')
    CLASS_ID = -1 
    for k, v in YOLO_MODEL.names.items():
        if v == 'dog':
            CLASS_ID = k
            break
except Exception as e:
    print(f"Error al cargar el modelo YOLO: {e}")
    YOLO_MODEL = None

def detect_bounding_boxes(
    source, 
    imgsz = 640, 
    conf_threshold = 0.5, 
    save_image = False):
    """
    Detecta perros en una imagen dada y devuelve una lista de sus bounding boxes.
    Acepta una imagen PIL o array de NumPy como fuente.
    """
    if YOLO_MODEL is None or CLASS_ID is None:
        print("El modelo YOLO no está disponible.")
        return []

    results = YOLO_MODEL.predict(
        source=source,
        imgsz=imgsz,
        conf=conf_threshold,
        save=save_image,
        verbose=False
    )

    detections = []
    
    if results:
        r = results[0]
        boxes = r.boxes[r.boxes.cls == CLASS_ID].cpu().numpy()
        
        for det in boxes:
            # Coordenadas en formato xyxy
            x1, y1, x2, y2 = det.xyxy[0].astype(int) 
            conf = det.conf[0]
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': round(conf, 4)
            })

    return detections

# --- Clasificador por Similitud ---

OUTPUT_DIR = 'data/processed'
INDEX_R50_FILENAME = 'faiss_index_R50.bin'
PATHS_FILENAME = 'image_paths.npy'
K_SIMILAR = 10

FAISS_PATH_R50 = os.path.join(OUTPUT_DIR, INDEX_R50_FILENAME)
PATHS_PATH = os.path.join(OUTPUT_DIR, PATHS_FILENAME)

MODELS = {} 
FAISS_INDEXES = {}
IMAGE_PATHS = None
SYSTEM_READY = False
MODEL_KEY = 'ResNet50'

def load_and_preprocess_image_keras(img_input, target_size=(224, 224)):
    """Pre-procesa una imagen para el modelo Keras (ResNet50)."""
    img = img_input.resize(target_size)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def get_breed_from_path(path):
    """Extrae el nombre de la raza."""
    return os.path.basename(os.path.dirname(path)).replace('_', ' ')

def initialize_search_system():
    """Carga solo ResNet50 y su índice FAISS."""
    global MODELS, FAISS_INDEXES, IMAGE_PATHS, SYSTEM_READY
    
    print("Iniciando el sistema de búsqueda (Solo ResNet50)...")
    SYSTEM_READY = False
    
    if not os.path.exists(PATHS_PATH):
        print(f"FALLO CRÍTICO: No se encontraron las rutas de imagen en {PATHS_PATH}.")
        return
    IMAGE_PATHS = np.load(PATHS_PATH)
    
    try:
        print("Intentando cargar ResNet50 de Keras...")
        if not os.path.exists(FAISS_PATH_R50):
            print(f"FALLO CRÍTICO: Faltan índices FAISS R50: {FAISS_PATH_R50}.")
        else:
            MODELS[MODEL_KEY] = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            FAISS_INDEXES[MODEL_KEY] = faiss.read_index(FAISS_PATH_R50)
            print("✅ Modelo ResNet50 cargado exitosamente.")
        
    except Exception as e:
        print(f"Falló la carga del Modelo ResNet50. Detalle: {e}")
        
    if not MODELS:
        print("Error: No se pudo cargar ResNet50.")
        return

    print(f"Sistema inicializado. {len(MODELS)} modelos disponibles: {list(MODELS.keys())}.")
    SYSTEM_READY = True

initialize_search_system()

def get_embedding(model_key, input_img_pil):
    """Extrae el vector de features usando el modelo ResNet50 (Keras)."""
    if model_key != MODEL_KEY:
        raise ValueError(f"Modelo {model_key} no soportado. Solo se soporta {MODEL_KEY}.")

    model = MODELS[MODEL_KEY]
    
    processed_img = load_and_preprocess_image_keras(input_img_pil)
    embedding = model.predict(processed_img, verbose=0).flatten().astype('float32')
        
    return np.expand_dims(embedding, axis=0)

def classify_dog_breed(cropped_img_pil):
    """
    Clasifica un perro recortado utilizando el modelo ResNet50 de búsqueda por similitud.
    """
    if not SYSTEM_READY or MODEL_KEY not in MODELS:
        return "ERROR: Sistema no inicializado.", None

    faiss_index = FAISS_INDEXES[MODEL_KEY]
    query_embedding = get_embedding(MODEL_KEY, cropped_img_pil)

    # Búsqueda con FAISS
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
    
    return predicted_breed, retrieved_breeds

def integrated_detection_and_classification(input_image_pil):
    """
    Flujo de trabajo completo:
    1. Detecta bounding boxes de perros.
    2. Recorta cada perro detectado.
    3. Clasifica la raza de cada recorte (usando ResNet50 por defecto).
    4. Dibuja los boxes y etiquetas sobre la imagen original.
    """
    if not SYSTEM_READY or MODEL_KEY not in MODELS:
        error_msg = f"Error: Modelo '{MODEL_KEY}' no disponible o sistema no inicializado."
        return input_image_pil, [], error_msg

    # Conversión a NumPy para robustez con YOLO.predict()
    source_array = np.array(input_image_pil)
    
    # 1. Detección de perros
    detections = detect_bounding_boxes(source_array)
    
    if not detections:
        return input_image_pil, [], "No se detectaron perros en la imagen."

    # Preparar la imagen para dibujar
    img_draw = input_image_pil.copy()
    draw = ImageDraw.Draw(img_draw)

    processed_results = []
    
    # 2. y 3. Recorte y Clasificación
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        
        # 3. Recortar la imagen
        cropped_dog = input_image_pil.crop((x1, y1, x2, y2))
        
        # 4. Clasificar la raza (ya no necesita el parámetro de selección de modelo)
        predicted_breed, _ = classify_dog_breed(cropped_dog)

        # 5. Dibujar bounding box y etiqueta
        color = (255, 0, 0)
        label = f"{predicted_breed} ({conf:.2f})"
        
        # Dibujar el rectángulo
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        
        # Dibujar el fondo del texto y el texto
        try:
            text_bbox = draw.textbbox((x1, y1 - 22), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill=color)
            draw.text((x1 + 2, y1 - text_height), label, fill=(255, 255, 255))
        except Exception:
             draw.text((x1 + 2, y1 - 20), label, fill=(255, 255, 255))

        # Acumular resultados para mostrar en la galería
        processed_results.append((cropped_dog.resize((224, 224)), f"Raza: {predicted_breed}\nConfianza YOLO: {conf:.2f}"))

    # Resultado final de la imagen
    final_image = img_draw
    
    # Texto de resumen
    summary_msg = f"Detectados y clasificados {len(detections)} perros utilizando el modelo '{MODEL_KEY}'."

    return final_image, processed_results, summary_msg


# --- Interfaz Gradio ---

if SYSTEM_READY and YOLO_MODEL is not None:
    
    gallery_output = gr.Gallery(
        label="Recortes y Predicciones de Perros Detectados", 
        columns=5, 
        rows=2, 
        object_fit="contain",
        height=450
    )
    
    with gr.Blocks(title="Clasificador Integrado (YOLO + ResNet50)") as app:
        gr.Markdown(
            f"# 🐕 Clasificador Integrado de Razas de Perro (YOLO + {MODEL_KEY})\n"
            "El sistema detecta perros, recorta la imagen y clasifica la raza usando **ResNet50**."
        )

        predicted_output = gr.Textbox(
            label="Resumen de la Detección y Clasificación", 
            value="Esperando imagen...", 
            scale=1
        )

        with gr.Row():
            # Entrada
            input_image = gr.Image(type="pil", label="Imagen de Entrada", width=400)
            
            # Botón de búsqueda
            process_button = gr.Button(f"Detectar y Clasificar Razas (Usando {MODEL_KEY})", scale=1)
        
        with gr.Row():
            # Salida: Imagen original con boxes
            output_image_with_boxes = gr.Image(
                type="pil", 
                label="Imagen con Bounding Boxes y Raza Predicha", 
                width=600
            )
            
            # Galería de recortes
            gallery_output.render()

        # Conectar el botón con la función
        process_button.click(
            fn=integrated_detection_and_classification,
            inputs=[input_image],
            outputs=[output_image_with_boxes, gallery_output, predicted_output]
        )
        
    # Lanzar la aplicación
    app.launch()
else:
    print("No se pudo lanzar la aplicación.")