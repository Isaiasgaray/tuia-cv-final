from ultralytics import YOLO
import os
import numpy as np
import faiss
from PIL import Image, ImageDraw
from collections import Counter
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models, transforms

OUTPUT_DIR = 'data/processed'
INDEX_R18_FILENAME = 'faiss_index_R18.bin'
PATHS_FILENAME = 'image_paths.npy'
K_SIMILAR = 10
FAISS_PATH_R18 = os.path.join(OUTPUT_DIR, INDEX_R18_FILENAME)
PATHS_PATH = os.path.join(OUTPUT_DIR, PATHS_FILENAME)
MODEL_KEY = 'ResNet18'
WEIGHTS_R18_PATH = 'data/resnet18_70_breeds_best_weights.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASS_ID = -1 

# --- Carga de Modelos ---
YOLO_MODEL = None
MODELS = {} 
FAISS_INDEXES = {}
IMAGE_PATHS = None
SYSTEM_READY = False

try:
    YOLO_MODEL = YOLO('yolov8n.pt')
    for k, v in YOLO_MODEL.names.items():
        if v == 'dog':
            CLASS_ID = k
            break
except Exception as e:
    print(f"Error al cargar el modelo YOLOv8n: {e}")

# --- Funciones de Inicialización ---

def get_pytorch_transforms():
    """Define las transformaciones estándar para el modelo PyTorch."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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

def get_breed_from_path(path):
    """Extrae el nombre de la raza."""
    return os.path.basename(os.path.dirname(path)).replace('_', ' ')

def initialize_search_system():
    """Carga ResNet18 y su índice FAISS."""
    global MODELS, FAISS_INDEXES, IMAGE_PATHS, SYSTEM_READY
    
    if not os.path.exists(PATHS_PATH):
        print(f"FALLO CRÍTICO: No se encontraron las rutas de imagen en {PATHS_PATH}. Saltando la carga de FAISS.")
        return

    IMAGE_PATHS = np.load(PATHS_PATH)
    
    try:
        print("Intentando cargar ResNet18...")
        if not os.path.exists(FAISS_PATH_R18):
            print(f"FALLO CRÍTICO: Faltan índices FAISS R18: {FAISS_PATH_R18}.")
        else:
            MODELS['ResNet18'] = load_pytorch_resnet18()
            FAISS_INDEXES['ResNet18'] = faiss.read_index(FAISS_PATH_R18)
            print("✅ Modelo ResNet18 cargado exitosamente.")
        
    except Exception as e:
        print(f"Falló la carga del Modelo ResNet18. Detalle: {e}")
        
    if not MODELS:
        print("Error: No se pudo cargar ResNet18.")
        return

    print(f"Sistema inicializado. {len(MODELS)} modelos disponibles: {list(MODELS.keys())}.")
    SYSTEM_READY = True

initialize_search_system()

# --- Funciones de Detección ---

def detect_bounding_boxes(source, conf_threshold):
    """Detecta perros en una imagen y devuelve boxes y confianza."""
    if YOLO_MODEL is None or CLASS_ID is None:
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
                'class': 'dog'
            })

    return detections

# --- Funciones de Clasificación ---

def get_embedding(model_key, input_img_pil):
    """Extrae el vector de features usando el modelo ResNet18."""
    if model_key != MODEL_KEY:
        raise ValueError(f"Modelo {model_key} no soportado. Solo se soporta {MODEL_KEY}.")

    model = MODELS[MODEL_KEY]
    
    preprocess = get_pytorch_transforms()
    input_tensor = preprocess(input_img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    embedding = output.cpu().numpy().astype('float32')
    
    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)
    elif embedding.ndim > 2 or embedding.shape[0] != 1:
        embedding = embedding.flatten().reshape(1, -1)
        
    return embedding

def classify_dog_breed(cropped_img_pil):
    """Clasifica un perro recortado utilizando el modelo ResNet18 de búsqueda por similitud."""
    if not SYSTEM_READY or MODEL_KEY not in MODELS:
        return "ERROR: Sistema no inicializado.", []

    faiss_index = FAISS_INDEXES[MODEL_KEY]
    try:
        query_embedding = get_embedding(MODEL_KEY, cropped_img_pil)
    except ValueError as e:
         print(f"Error al obtener embedding: {e}")
         return "ERROR: Fallo de embedding", []

    D, I = faiss_index.search(query_embedding, K_SIMILAR + 1)

    start_index = 0
    if D[0][0] < 1e-6:
        start_index = 1
    
    result_indices = I[0][start_index:K_SIMILAR + start_index]
    similar_image_paths = IMAGE_PATHS[result_indices]

    retrieved_breeds = [get_breed_from_path(path) for path in similar_image_paths]
    breed_counts = Counter(retrieved_breeds)
    predicted_breed = breed_counts.most_common(1)[0][0]
    
    return predicted_breed, retrieved_breeds

# Conjunto de Prueba (Ground Truth)

TEST_IMAGES_DIR = 'data/dogs' 
LABELS_DIR = os.path.join(TEST_IMAGES_DIR, 'labels')
LABELS_TXT = os.path.join(LABELS_DIR, 'labels.txt')

ID_TO_BREED = dict()

# Obtiene los labels y clases del archivo
# del etiquetado manual
with open (LABELS_TXT, "r") as f:
    for line in f:
        id, breed = line.strip().split(",")
        ID_TO_BREED[int(id)] = breed

def load_gt_data_from_yolo_multi_breed(image_dir, label_dir, id_to_breed_map):
    """
    Carga los datos de Ground Truth desde archivos YOLO (.txt) soportando 
    múltiples razas (clases) por imagen.
    """
    gt_data = []
    
    def get_image_size(path):
        try:
            with Image.open(path) as img:
                return img.width, img.height
        except Exception:
            return None, None

    # Iterar sobre todos los archivos de etiquetas .txt
    for label_filename in os.listdir(label_dir):
        if not label_filename.endswith('.txt'):
            continue
            
        base_name = label_filename.replace('.txt', '')
        image_filename = base_name + '.jpg'
        image_path = os.path.join(image_dir, image_filename)
        label_path = os.path.join(label_dir, label_filename)
        
        W, H = get_image_size(image_path)
        if W is None:
            continue
        
        current_gt_boxes = []

        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                class_id = int(class_id)
                
                # Obtener la raza usando el ID
                gt_breed = id_to_breed_map.get(class_id)
                
                if gt_breed is None:
                    continue

                # Desnormalizar coordenadas a píxeles
                x_center *= W
                y_center *= H
                width *= W
                height *= H
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Formato requerido: [x1, y1, x2, y2, 'raza_correcta']
                current_gt_boxes.append([x1, y1, x2, y2, gt_breed])
        
        if current_gt_boxes:
            gt_data.append({
                'path': image_path,
                'gt_boxes': current_gt_boxes
            })

    return gt_data

gt_data = load_gt_data_from_yolo_multi_breed(TEST_IMAGES_DIR, LABELS_DIR, ID_TO_BREED)

# Total de instancias de perros reales para el Recall:
TOTAL_GT_INSTANCES = sum(len(item['gt_boxes']) for item in gt_data)
print(f"✅ GT data cargado automáticamente de {len(gt_data)} imágenes.")
print(f"Total de instancias de GT (Ground Truth) a evaluar: {TOTAL_GT_INSTANCES}")

# Función de Evaluación y Métricas

def draw_bounding_boxes(
    img_pil, 
    gt_boxes, 
    pred_boxes,
    gt_color = 'green',
    pred_color = 'red',
    box_width = 4):
    """
    Dibuja las cajas de Ground Truth y Predicción sobre una imagen PIL.
    """
    draw = ImageDraw.Draw(img_pil)
    
    # --- Dibujar Cajas de Ground Truth (GT) ---
    for box_data in gt_boxes:
        x1, y1, x2, y2, breed = box_data
        # Dibujar el rectángulo
        draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=box_width)
        
        text = "Ground Truth - " + breed
        font_size = 30
        draw.rectangle([x1, y1 - font_size - 5, x1 + 150, y1], fill=gt_color)
        draw.text((x1 + 5, y1 - font_size), text, fill='white')

    # --- Dibujar Cajas de Predicción (PR) ---
    for det in pred_boxes:
        x1, y1, x2, y2 = det['box']
        
        # Dibujar el rectángulo
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=box_width)

        predicted_breed, _ = classify_dog_breed(img_pil.crop((x1, y1, x2, y2)))

        text = "Prediction - " + predicted_breed
        font_size = 30 
        
        # Dibuja la etiqueta en la parte inferior o superior
        text_x, text_y = x2 - 150, y2
        
        draw.rectangle([text_x, text_y, text_x + 150, text_y + font_size + 5], fill=pred_color)
        draw.text((text_x + 5, text_y + 5), text, fill='white')

    return img_pil


def evaluate_pipeline(gt_data):
    """
    Ejecuta el pipeline en los datos de prueba y genera:
    1. Un DataFrame de detecciones para métricas de Detección.
    2. Una lista para métricas de Clasificación (usando recortes GT).
    """
    
    # 1. Detección: DataFrame para mAP/Precision/Recall 
    # Formato: [image_id, class_name, confidence, x1, y1, x2, y2]
    detection_results = []
    
    # 2. Clasificación: Lista para el Accuracy
    classification_results = []
    
    for image_idx, gt_item in enumerate(gt_data):
        img_path = gt_item['path']
        
        try:
            input_image_pil = Image.open(img_path).convert("RGB")
            source_array = np.array(input_image_pil)
        except Exception as e:
            print(f"Error al abrir la imagen {img_path}: {e}")
            continue

        # --- Ejecutar Detección ---
        detections = detect_bounding_boxes(source_array, conf_threshold=0.5)
        # predicted_breed, _ = classify_dog_breed(input_image_pil)
        # print(predicted_breed)

        # Dibuja las cajas en una copia de la imagen PIL
        # print(detections)
        img_with_boxes = draw_bounding_boxes(
            input_image_pil.copy(), 
            gt_item['gt_boxes'], 
            detections)
        
        # Guardar la imagen visualizada
        output_vis_dir = 'data/vis_results'
        if not os.path.exists(output_vis_dir):
            os.makedirs(output_vis_dir)
        vis_path = os.path.join(output_vis_dir, os.path.basename(img_path).replace('.jpg', '_vis.jpg'))
        img_with_boxes.save(vis_path)
        
        # --- Clasificación y Recolección de Predicciones ---
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['conf']
            
            # Recortar y Clasificar solo si las coordenadas son válidas
            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                cropped_dog = input_image_pil.crop((x1, y1, x2, y2))
                predicted_breed, _ = classify_dog_breed(cropped_dog)
            else:
                predicted_breed = "UNKNOWN_INVALID_BOX"
            
            # Recolección para Detección (DataFrame)
            detection_results.append({
                'image_id': image_idx,
                'class_name': 'dog',
                'conf': conf,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'predicted_breed': predicted_breed 
            })

        # Recolección de Ground Truth para Evaluación de Clasificación (Accuracy)
        # Se evalúa la precisión del clasificador usando los recortes perfectos de GT.
        
        for gt_box in gt_item['gt_boxes']:
            gt_x1, gt_y1, gt_x2, gt_y2, gt_breed = gt_box
            
            # Recortar usando el Ground Truth Box
            cropped_gt = input_image_pil.crop((gt_x1, gt_y1, gt_x2, gt_y2))
            
            # Clasificar el recorte de Ground Truth
            predicted_breed_gt_crop, _ = classify_dog_breed(cropped_gt)

            classification_results.append({
                'gt_breed': gt_breed,
                'predicted_breed': predicted_breed_gt_crop
            })
            
    # Convertir resultados de detección a DataFrame
    df_detections = pd.DataFrame(detection_results)
    
    # Generar los archivos de Ground Truth en formato requerido por las librerías de mAP
    if not os.path.exists('eval_gt'):
         os.makedirs('eval_gt')
         
    for image_idx, gt_item in enumerate(gt_data):
        gt_filename = os.path.join('eval_gt', f'img_{image_idx}.txt')
        with open(gt_filename, 'w') as f:
            for gt_box in gt_item['gt_boxes']:
                x1, y1, x2, y2, breed = gt_box
                # Formato: class_name x1 y1 x2 y2
                f.write(f"dog {x1} {y1} {x2} {y2}\n")

    return df_detections, classification_results

# --- Ejecución ---
df_detections, classification_results = evaluate_pipeline(gt_data)
print(f"\nGenerado DataFrame de detecciones con {len(df_detections)} predicciones.")
print(f"Recolectados {len(classification_results)} resultados de clasificación GT para Accuracy.")

# Cálculo de Métricas de Detección (P, R, F1, mAP)

def calculate_iou(boxA, boxB):
    """Calcula IoU de dos bounding boxes [x1, y1, x2, y2]"""
    # 1. Determinar las coordenadas de intersección (over-lapping region)
    # xA, yA: esquina superior izquierda de la intersección
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    # xB, yB: esquina inferior derecha de la intersección
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 2. Calcular el área de intersección
    # Se asegura que la intersección sea positiva (si no hay solapamiento, el área es 0)
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 3. Calcular el área de las cajas individuales (Unión = AreaA + AreaB - Intersección)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 4. Calcular el IoU
    # La unión se calcula como (AreaA + AreaB - Intersección)
    unionArea = float(boxAArea + boxBArea - interArea)
    
    # Manejar el caso de división por cero
    iou = interArea / unionArea if unionArea > 0 else 0
    return iou

# --- Cálculo de P, R, F1 (Usando IoU = 0.5 y Conf. = 0.5) ---
IOU_THRESHOLD = 0.5
CONF_THRESHOLD_PR = 0.5 
# Verdaderos Positivos
tp = 0
# Falsos Positivos
fp = 0
matched_gt = set()

# Filtrar detecciones por umbral de confianza para P, R, F1
df_filtered_pr = df_detections[df_detections['conf'] >= CONF_THRESHOLD_PR].sort_values(by='conf', ascending=False)

# 1. Matching de Predicciones (PR) con Ground Truth (GT)
for _, pr_row in df_filtered_pr.iterrows():
    pr_box = [pr_row['x1'], pr_row['y1'], pr_row['x2'], pr_row['y2']]
    image_id = pr_row['image_id']
    
    current_gt_boxes = gt_data[image_id]['gt_boxes']
    best_iou = 0
    best_gt_idx = -1
    
    for gt_idx, gt_box in enumerate(current_gt_boxes):
        gt_identifier = (image_id, gt_idx) 
        
        # Solo comparar con GT boxes que aún no han sido matcheados
        if gt_identifier not in matched_gt:
            gt_coords = gt_box[:4]
            iou = calculate_iou(pr_box, gt_coords)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

    # 2. Contar TP/FP
    if best_iou >= IOU_THRESHOLD and (image_id, best_gt_idx) not in matched_gt:
        tp += 1
        matched_gt.add((image_id, best_gt_idx))
    else:
        fp += 1

# 3. Contar FN
fn = TOTAL_GT_INSTANCES - len(matched_gt)

# 4. Cálculo de Métricas Simples
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
# --- Cálculo de IoU Promedio ---
iou_scores_tp = []

# Mismo matching que se usó para P, R, F1, pero ahora almacenando los scores de IoU de los TP
matched_gt_iou = set()

for _, pr_row in df_filtered_pr.iterrows():
    pr_box = [pr_row['x1'], pr_row['y1'], pr_row['x2'], pr_row['y2']]
    image_id = pr_row['image_id']
    
    current_gt_boxes = gt_data[image_id]['gt_boxes']
    best_iou = 0
    best_gt_idx = -1
    
    for gt_idx, gt_box in enumerate(current_gt_boxes):
        gt_identifier = (image_id, gt_idx) 
        
        # Solo comparar con GT boxes que aún no han sido matcheados
        if gt_identifier not in matched_gt_iou:
            gt_coords = gt_box[:4]
            iou = calculate_iou(pr_box, gt_coords)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

    if best_iou >= IOU_THRESHOLD and (image_id, best_gt_idx) not in matched_gt_iou:
        # Si es un Verdadero Positivo, guardamos su score de IoU
        iou_scores_tp.append(best_iou)
        matched_gt_iou.add((image_id, best_gt_idx))
        
# Calcular el IoU promedio de los Verdaderos Positivos (TP)
mean_iou = np.mean(iou_scores_tp) if iou_scores_tp else 0

# MEDICIÓN DE PRECISIÓN (mAP)
def medir_precision(modelo, dataset="coco128.yaml"):
    resultados = modelo.val(data=dataset)
    return resultados.results_dict.get("metrics/mAP50-95(B)", None)

mAP_50 = medir_precision(YOLO_MODEL)

print("\n--- Resultados de Evaluación de Detección (YOLO) ---")
print(f"Total de instancias GT: {TOTAL_GT_INSTANCES}")
print(f"Umbral IoU usado: {IOU_THRESHOLD}, Umbral Conf.: {CONF_THRESHOLD_PR}")
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Falsos Negativos (FN): {fn}")
print("-------------------------------------------------")
print(f"IoU Promedio (TP): {mean_iou:.4f}")
print(f"Precisión (P): {precision:.4f}")
print(f"Recall (R): {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"mAP@0.5: {mAP_50:.4f}") 

# Métrica de Clasificación (Accuracy)

correct_predictions = 0
total_classification = len(classification_results)

for res in classification_results:
    # Se comparan las razas en minúsculas para evitar errores de capitalización
    if res['gt_breed'].lower() == res['predicted_breed'].lower():
        correct_predictions += 1

accuracy = correct_predictions / total_classification if total_classification > 0 else 0

print("\n--- Resultados de Evaluación de Clasificación (ResNet18/FAISS) ---")
print(f"Total de instancias clasificadas (Recortes GT): {total_classification}")
print(f"Clasificaciones Correctas: {correct_predictions}")
print(f"Accuracy de Clasificación (Raza): {accuracy:.4f}")