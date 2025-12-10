import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import faiss
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from collections import Counter

# Definiciones y Carga de Componentes

# --- Constantes de Configuración ---
INPUT_IMAGES_DIR = 'data/dogs' 
OUTPUT_ANNOTATIONS_DIR = 'data/auto_annotations'
# Umbral de confianza para la detección
CONFIDENCE_THRESHOLD = 0.5
# Parámetro K para la búsqueda FAISS
K_SIMILAR = 10

# Configuración del modelo FAISS
OUTPUT_DIR = 'data/processed'
INDEX_R50_FILENAME = 'faiss_index_R50.bin'
PATHS_FILENAME = 'image_paths.npy'
FAISS_PATH_R50 = os.path.join(OUTPUT_DIR, INDEX_R50_FILENAME)
PATHS_PATH = os.path.join(OUTPUT_DIR, PATHS_FILENAME)
MODEL_KEY = 'ResNet50'
CLASS_ID = -1

YOLO_MODEL = None
MODELS = {}
FAISS_INDEXES = {}
IMAGE_PATHS = None
SYSTEM_READY = False
# Mapeo final de ID único a nombre de raza para el formato COCO
ID_TO_BREED = {}
# Mapeo de nombre de raza a ID único
BREED_TO_ID = {}

# --- Funciones de Inicialización ---

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
    """Carga YOLOv8n, ResNet50 y su índice FAISS."""
    global YOLO_MODEL, MODELS, FAISS_INDEXES, IMAGE_PATHS, SYSTEM_READY, CLASS_ID
    global ID_TO_BREED, BREED_TO_ID

    # Cargar YOLO
    try:
        YOLO_MODEL = YOLO('yolov8n.pt')
        for k, v in YOLO_MODEL.names.items():
            if v == 'dog':
                CLASS_ID = k
                break
        if CLASS_ID == -1:
            print("FALLO: La clase 'dog' no se encontró en el modelo YOLO.")
            return
    except Exception as e:
        print(f"Error al cargar el modelo YOLOv8n: {e}")
        return

    # Cargar Rutas e Índices FAISS (para ResNet50)
    if not os.path.exists(PATHS_PATH):
        print(f"FALLO CRÍTICO: No se encontraron las rutas de imagen en {PATHS_PATH}. Saltando la carga de FAISS.")
        return

    IMAGE_PATHS = np.load(PATHS_PATH)

    try:
        if not os.path.exists(FAISS_PATH_R50):
            print(f"FALLO CRÍTICO: Faltan índices FAISS R50: {FAISS_PATH_R50}. Saltando la carga.")
            return

        MODELS[MODEL_KEY] = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        FAISS_INDEXES[MODEL_KEY] = faiss.read_index(FAISS_PATH_R50)
        print("✅ ResNet50 y FAISS cargados exitosamente.")

        # Crear el mapa de IDs para COCO
        # Obtener todas las razas únicas del dataset
        all_breeds = sorted(list(set(get_breed_from_path(path) for path in IMAGE_PATHS)))
        ID_TO_BREED = {i + 1: breed for i, breed in enumerate(all_breeds)}
        BREED_TO_ID = {breed: i + 1 for i, breed in enumerate(all_breeds)}

        SYSTEM_READY = True

    except Exception as e:
        print(f"Falló la carga del Modelo ResNet50 o FAISS. Detalle: {e}")

# --- Funciones de Detección ---

def detect_bounding_boxes(source, conf_threshold):
    """Detecta perros en una imagen y devuelve boxes y confianza."""
    if YOLO_MODEL is None or CLASS_ID == -1:
        return []

    results = YOLO_MODEL.predict(
        source=source,
        conf=conf_threshold,
        verbose=False
    )

    detections = []
    if results:
        r = results[0]
        # Filtrar solo la clase 'dog'
        boxes = r.boxes[r.boxes.cls == CLASS_ID].cpu().numpy()

        for det in boxes:
            # Coordenadas en formato xyxy
            x1, y1, x2, y2 = det.xyxy[0].astype(int)
            conf = det.conf[0]

            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': round(conf, 4),
                # Clase genérica de detección
                'class': 'dog'
            })
    return detections

# --- Funciones de Clasificación ---

def get_embedding(model_key, input_img_pil):
    """Extrae el vector de features usando el modelo ResNet50 (Keras)."""
    if model_key != MODEL_KEY or MODEL_KEY not in MODELS:
        raise ValueError("Modelo no disponible o no soportado.")

    model = MODELS[MODEL_KEY]
    processed_img = load_and_preprocess_image_keras(input_img_pil)
    embedding = model.predict(processed_img, verbose=0).flatten().astype('float32')
    return np.expand_dims(embedding, axis=0)

def classify_dog_breed(cropped_img_pil):
    """Clasifica un perro recortado utilizando el modelo ResNet50 de búsqueda por similitud."""
    if not SYSTEM_READY or MODEL_KEY not in MODELS:
        return "ERROR_SYSTEM_NOT_READY", []

    faiss_index = FAISS_INDEXES[MODEL_KEY]
    try:
        query_embedding = get_embedding(MODEL_KEY, cropped_img_pil)
    except ValueError as e:
        return "ERROR_EMBEDDING_FAILURE", []

    D, I = faiss_index.search(query_embedding, K_SIMILAR + 1)

    start_index = 0
    if D[0][0] < 1e-6:
        start_index = 1

    result_indices = I[0][start_index:K_SIMILAR + start_index]
    similar_image_paths = IMAGE_PATHS[result_indices]

    retrieved_breeds = [get_breed_from_path(path) for path in similar_image_paths]
    breed_counts = Counter(retrieved_breeds)
    # Selecciona la raza más común
    predicted_breed = breed_counts.most_common(1)[0][0]

    return predicted_breed, retrieved_breeds



# Funciones de Conversión de Formato (YOLOv5 y COCO)

def normalize_box_yolo(x1, y1, x2, y2, W, H):
    """Convierte coordenadas absolutas (x1, y1, x2, y2) a formato YOLO (xc, yc, w, h) normalizado."""
    if W == 0 or H == 0:
        return None
    x_center = ((x1 + x2) / 2) / W
    y_center = ((y1 + y2) / 2) / H
    width = (x2 - x1) / W
    height = (y2 - y1) / H
    return [x_center, y_center, width, height]

def generate_yolov5_annotation(detections, W, H):
    """Genera el contenido para el archivo .txt de YOLOv5."""
    yolo_lines = []
    for det in detections:
        x1, y1, x2, y2 = det['box']
        predicted_breed = det['breed']

        norm_box = normalize_box_yolo(x1, y1, x2, y2, W, H)
        if norm_box is None:
            continue

        breed_id = BREED_TO_ID.get(predicted_breed)
        if breed_id is not None:
            # Formato: <class_id> <x_center> <y_center> <width> <height>
            line = f"{breed_id} {norm_box[0]:.6f} {norm_box[1]:.6f} {norm_box[2]:.6f} {norm_box[3]:.6f}"
            yolo_lines.append(line)
    return "\n".join(yolo_lines)

def create_coco_annotation(image_id, file_name, W, H, all_detections):
    """Genera la estructura de anotación COCO para una sola imagen."""
    coco_image = {
        "id": int(image_id),
        "file_name": file_name,
        "width": int(W),
        "height": int(H)
    }
    coco_annotations = []
    annotation_id_counter = 1

    for det in all_detections:
        x1, y1, x2, y2 = det['box']
        predicted_breed = det['breed']
        conf = det['conf']

        width = x2 - x1
        height = y2 - y1

        # Formato COCO bbox: [x_min, y_min, width, height]
        bbox_coco = [x1, y1, width, height]
        breed_id = BREED_TO_ID.get(predicted_breed)

        if breed_id is not None:
            coco_annotations.append({
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": breed_id,
                "bbox": [int(n) for n in bbox_coco],
                "area": float(width * height),
                "iscrowd": 0,
                "confidence": float(conf),
                "class_name": predicted_breed
            })
            annotation_id_counter += 1

    return coco_image, coco_annotations

def create_coco_categories():
    """Genera la lista de categorías para el archivo COCO, basada en las razas."""
    return [
        {"id": breed_id, "name": breed_name, "supercategory": "dog_breed"}
        for breed_id, breed_name in ID_TO_BREED.items()
    ]


# Pipeline de Anotación

def run_annotation_pipeline(input_dir, output_dir, conf_threshold):
    """
    Ejecuta el pipeline de detección + clasificación para anotar todas las imágenes
    en el directorio de entrada y guardar los resultados en los formatos requeridos.
    """
    if not SYSTEM_READY:
        print("El sistema no está inicializado correctamente. Abortando el pipeline.")
        return

    # Crear directorios de salida
    yolo_output_dir = os.path.join(output_dir, 'yolov5_labels')
    coco_output_file = os.path.join(output_dir, 'coco_annotations.json')
    os.makedirs(yolo_output_dir, exist_ok=True)

    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": create_coco_categories(),
        "images": [],
        "annotations": []
    }

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_id_counter = 1
    # Para las anotaciones COCO
    annotation_id_tracker = 1

    print(f"Iniciando anotación de {len(image_files)} imágenes...")

    for file_name in tqdm(image_files, desc="Procesando imágenes"):
        img_path = os.path.join(input_dir, file_name)

        try:
            input_image_pil = Image.open(img_path).convert("RGB")
            W, H = input_image_pil.size
            source_array = np.array(input_image_pil)
        except Exception as e:
            print(f"Error al abrir la imagen {img_path}: {e}")
            continue

        # Detección (YOLO)
        detections = detect_bounding_boxes(source_array, conf_threshold=conf_threshold)

        # Clasificación (ResNet50)
        final_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            # Recortar y Clasificar solo si las coordenadas son válidas
            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                cropped_dog = input_image_pil.crop((x1, y1, x2, y2))
                predicted_breed, _ = classify_dog_breed(cropped_dog)
            else:
                predicted_breed = "UNKNOWN_INVALID_BOX"

            if predicted_breed not in ("ERROR_SYSTEM_NOT_READY", "ERROR_EMBEDDING_FAILURE", "UNKNOWN_INVALID_BOX"):
                det['breed'] = predicted_breed
                final_detections.append(det)

        # Generar Anotación YOLOv5 (.txt)
        yolo_content = generate_yolov5_annotation(final_detections, W, H)
        yolo_filename = file_name.replace(os.path.splitext(file_name)[1], '.txt')
        yolo_path = os.path.join(yolo_output_dir, yolo_filename)
        with open(yolo_path, 'w') as f:
            f.write(yolo_content)

        # Generar Anotación COCO (.json)
        coco_image, coco_annotations = create_coco_annotation(
            image_id_counter, file_name, W, H, final_detections
        )

        coco_dataset['images'].append(coco_image)
        for ann in coco_annotations:
            ann["id"] = annotation_id_tracker
            coco_dataset['annotations'].append(ann)
            annotation_id_tracker += 1

        image_id_counter += 1

    # Guardar el archivo COCO completo
    with open(coco_output_file, 'w') as f:
       json.dump(coco_dataset, f)

    print("\nProceso de anotación automática finalizado.")
    print(f"Anotaciones YOLOv5 (.txt) guardadas en: {yolo_output_dir}")
    print(f"Anotaciones COCO (.json) guardadas en: {coco_output_file}")
    
initialize_search_system()
run_annotation_pipeline(INPUT_IMAGES_DIR, OUTPUT_ANNOTATIONS_DIR, CONFIDENCE_THRESHOLD)