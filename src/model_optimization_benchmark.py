import os
import time
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
import faiss

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
import torch.quantization as tq

# Intentamos importar ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception as e:
    print("Aviso: ultralytics no disponible:", e)
    HAS_ULTRALYTICS = False

# Intentamos importar onnx / onnxruntime para cuantizar ONNX
HAS_ONNX = False
HAS_ONNXRUNTIME = False
HAS_ONNX_QUANT = False
try:
    import onnx
    HAS_ONNX = True
except Exception as e:
    print("Aviso: onnx no disponible:", e)

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except Exception as e:
    print("Aviso: onnxruntime no disponible:", e)

try:
    # onnxruntime.quantization may be available as:
    # from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_ONNX_QUANT = True
except Exception as e:
    print("Aviso: onnxruntime.quantization no disponible:", e)


# CONFIGURACIÓN
OUTPUT_DIR = 'data/processed'
INDEX_R18_FILENAME = 'faiss_index_R18.bin'
PATHS_FILENAME = 'image_paths.npy'
K_SIMILAR = 10

FAISS_PATH_R18 = os.path.join(OUTPUT_DIR, INDEX_R18_FILENAME)
PATHS_PATH = os.path.join(OUTPUT_DIR, PATHS_FILENAME)

MODEL_KEY = 'ResNet18'
CLASS_ID = -1

YOLO_WEIGHTS = 'yolov8n.pt'
YOLO_ONNX = 'yolov8n.onnx'
YOLO_ONNX_INT8 = 'yolov8n_int8.onnx'

# Dataset de prueba
TEST_IMAGES_DIR = 'data/dogs'
LABELS_DIR = os.path.join(TEST_IMAGES_DIR, 'labels')
LABELS_TXT = os.path.join(LABELS_DIR, 'labels.txt')

# TRANSFORMS ResNet
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

transform_resnet18 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# CARGA/INICIALIZACIÓN MODELOS
MODELS = {}
FAISS_INDEXES = {}
IMAGE_PATHS = None
SYSTEM_READY = False

# Cargamos YOLO original
YOLO_MODEL_ORIGINAL = None
YOLO_MODEL_QUANTIZED_ONNX = None

if HAS_ULTRALYTICS:
    try:
        print("Cargando YOLO (ultralytics) - original...")
        YOLO_MODEL_ORIGINAL = YOLO(YOLO_WEIGHTS)
        # detectamos id de la clase 'dog' si existe
        for k, v in YOLO_MODEL_ORIGINAL.names.items():
            if v == 'dog':
                CLASS_ID = k
                break
        print("YOLO cargado; class_id(dog) =", CLASS_ID)
    except Exception as e:
        print("Error cargando YOLO original:", e)
        YOLO_MODEL_ORIGINAL = None


def initialize_search_system():
    global MODELS, FAISS_INDEXES, IMAGE_PATHS, SYSTEM_READY

    if not os.path.exists(PATHS_PATH):
        print(f"ERROR: No existe {PATHS_PATH}; asegúrate de crear anteriores pasos (FAISS y paths).")
        return

    IMAGE_PATHS = np.load(PATHS_PATH)

    if not os.path.exists(FAISS_PATH_R18):
        print(f"ERROR: No existe el índice FAISS en {FAISS_PATH_R18}")
        return

    try:
        print("Cargando ResNet18 (FP32) ...")
        model = resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Identity()
        model.eval().to(torch_device)

        MODELS[MODEL_KEY] = model
        FAISS_INDEXES[MODEL_KEY] = faiss.read_index(FAISS_PATH_R18)

        SYSTEM_READY = True
        print("✅ ResNet18 + FAISS cargado exitosamente")

    except Exception as e:
        print("Error cargando ResNet18/FAISS:", e)
        SYSTEM_READY = False

initialize_search_system()


# Cuantización ResNet18
def quantize_resnet_dynamic(save_path=None):
    """
    Aplica quantize_dynamic a ResNet18 para convertir capas lineales a INT8.
    Devuelve el modelo cuantizado.
    """
    if MODEL_KEY not in MODELS:
        raise RuntimeError("ResNet18 no cargado")

    print("Creando copia del modelo para cuantizar...")
    model_fp32 = MODELS[MODEL_KEY].to('cpu').eval()

    print("Aplicando torch.quantization.quantize_dynamic() sobre capas Linear...")
    model_int8 = tq.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)

    if save_path:
        torch.save(model_int8.state_dict(), save_path)
        print(f"Modelo ResNet18 cuantizado guardado en: {save_path}")

    # guardamos en el dict
    MODELS[MODEL_KEY + "_INT8"] = model_int8
    return model_int8


# Export y cuantización ONNX de YOLO (INT8)
def try_quantize_yolo_to_onnx_int8(yolo_model, onnx_out=YOLO_ONNX, onnx_int8_out=YOLO_ONNX_INT8, img_size=640):
    """
    Intenta exportar el modelo YOLOv8 a ONNX y cuantizarlo con onnxruntime.quantize_dynamic.
    """
    if yolo_model is None:
        print("YOLO no disponible; no se intentará cuantizar.")
        return None

    if not HAS_ONNX:
        print("No se puede exportar/usar ONNX: paquete onnx no instalado.")
        return None
    if not HAS_ONNX_QUANT:
        print("No se puede cuantizar ONNX: onnxruntime.quantization no disponible.")
        return None

    try:
        print("Exportando YOLO a ONNX...")
        yolo_model.export(format="onnx", imgsz=img_size, simplify=True, opset=12)
        if os.path.exists(onnx_out):
            print("ONNX exportado:", onnx_out)
        else:
            candidates = [f for f in os.listdir('.') if f.endswith('.onnx')]
            if candidates:
                onnx_out = candidates[-1]
                print("ONNX exportado:", onnx_out)
            else:
                print("No se encontró el archivo ONNX exportado.")
                return None

        print("Cuantizando ONNX a INT8 con onnxruntime.quantization...")
        quantize_dynamic(onnx_out, onnx_int8_out, weight_type=QuantType.QInt8)
        if os.path.exists(onnx_int8_out):
            print("ONNX cuantizado creado:", onnx_int8_out)
            return onnx_int8_out
        else:
            print("La cuantización no produjo el archivo esperado.")
            return None

    except Exception as e:
        print("Fallo export/quantize YOLO ->", e)
        return None


# Extracción de embeddings
def get_embedding_real(model_key, input_img_pil, model_variant='FP32'):
    """
    Obtiene embedding real ejecutando el modelo,
    midiendo la latencia de inferencia.
    """
    if model_key not in MODELS:
        raise ValueError("Modelo ResNet18 no disponible")

    if model_variant == 'FP32':
        model = MODELS[MODEL_KEY]
    elif model_variant == 'INT8':
        model = MODELS.get(MODEL_KEY + "_INT8")
        if model is None:
            raise ValueError("No existe modelo INT8; aplica quantize_resnet_dynamic primero.")
    else:
        raise ValueError("model_variant desconocido")

    model_device = 'cpu' if next(model.parameters()).device.type == 'cpu' else torch_device

    img_tensor = transform_resnet18(input_img_pil).unsqueeze(0).to(model_device)

    start = time.perf_counter()
    with torch.no_grad():
        emb_t = model(img_tensor)
    end = time.perf_counter()

    # convertimos a numpy float32 para usar con FAISS
    emb = emb_t.cpu().numpy().flatten().astype('float32')

    latency_ms = (end - start) * 1000.0
    return emb.reshape(1, -1), latency_ms

# Clasificación por FAISS
def get_breed_from_path(path):
    return os.path.basename(os.path.dirname(path)).replace("_", " ")

def classify_dog_breed(cropped_img_pil, model_variant='FP32'):
    if not SYSTEM_READY:
        return "ERROR: sistema no listo", [], 0

    faiss_index = FAISS_INDEXES[MODEL_KEY]

    emb, lat = get_embedding_real(MODEL_KEY, cropped_img_pil, model_variant)

    start_f = time.perf_counter()
    D, I = faiss_index.search(emb, K_SIMILAR + 1)
    end_f = time.perf_counter()

    total_latency = lat + (end_f - start_f) * 1000.0

    result_indices = I[0][1:] if D[0][0] < 1e-6 else I[0][:K_SIMILAR]
    similar_paths = IMAGE_PATHS[result_indices]

    retrieved_breeds = [get_breed_from_path(p) for p in similar_paths]
    predicted_breed = Counter(retrieved_breeds).most_common(1)[0][0] if retrieved_breeds else "unknown"

    return predicted_breed, retrieved_breeds, total_latency


# DETECCIÓN
def detect_with_ultralytics(image_np, conf_threshold=0.5, yolo_model=None):
    """Usa ultralytics YOLO.predict (devuelve detecciones filtradas por clase 'dog')."""
    if yolo_model is None:
        raise ValueError("Se requiere un modelo ultralytics para detección")
    start = time.perf_counter()
    results = yolo_model.predict(source=image_np, conf=conf_threshold, verbose=False)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000.0

    detections = []
    r = results[0]
    try:
        boxes_all = r.boxes.cpu().numpy()
        for b in boxes_all:
            pass
    except Exception:
        pass

    # Recorremos r.boxes y filtramos por clase
    try:
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            if cls != CLASS_ID:
                continue
            xyxy = box.xyxy.cpu().numpy()[0].astype(int).tolist()
            conf = float(box.conf.cpu().numpy()[0])
            detections.append({'box': xyxy, 'conf': conf, 'class': 'dog'})
    except Exception as e:
        print("Warning: no se pudieron parsear boxes de ultralytics:", e)

    return detections, latency_ms


def detect_with_onnx(image_np, onnx_session, input_size=640, conf_thr=0.25):
    """
    Inferencia ONNX simple
    """
    if onnx_session is None:
        raise ValueError("Se requiere onnx_session")

    img = Image.fromarray(image_np).convert('RGB')
    w0, h0 = img.size

    r = min(input_size / w0, input_size / h0)
    new_w, new_h = int(w0 * r), int(h0 * r)
    resized = img.resize((new_w, new_h))
    canvas = Image.new('RGB', (input_size, input_size), (114, 114, 114))
    paste_x = (input_size - new_w) // 2
    paste_y = (input_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    arr = np.array(canvas).astype('float32') / 255.0
    inp = np.transpose(arr, (2, 0, 1))[None, :, :, :].astype('float32')

    input_name = onnx_session.get_inputs()[0].name

    start = time.perf_counter()
    outputs = onnx_session.run(None, {input_name: inp})
    end = time.perf_counter()
    latency_ms = (end - start) * 1000.0

    dets = []
    try:
        out = outputs[0]
        if out.ndim == 3:
            out = out[0]
        for row in out:
            conf_obj = float(row[4])
            if conf_obj < conf_thr:
                continue
            cls_prob = row[5:]
            cls = int(np.argmax(cls_prob))
            cls_conf = float(np.max(cls_prob))
            final_conf = conf_obj * cls_conf
            if final_conf < conf_thr:
                continue
            if cls != CLASS_ID:
                continue
            x, y, w, h = row[0:4]
            x1 = int((x - w/2) * input_size)
            y1 = int((y - h/2) * input_size)
            x2 = int((x + w/2) * input_size)
            y2 = int((y + h/2) * input_size)

            # Ajustar por padding y escala para mapear a la imagen original:
            # revertimos letterbox
            x1 = int((x1 - paste_x) / r)
            x2 = int((x2 - paste_x) / r)
            y1 = int((y1 - paste_y) / r)
            y2 = int((y2 - paste_y) / r)

            dets.append({'box': [x1, y1, x2, y2], 'conf': final_conf, 'class': 'dog'})

    except Exception as e:
        print("Warning: postprocess ONNX detections falló:", e)

    return dets, latency_ms



# CARGA GT
ID_TO_BREED = {}
if os.path.exists(LABELS_TXT):
    with open(LABELS_TXT, "r") as f:
        for line in f:
            id, breed = line.strip().split(",")
            ID_TO_BREED[int(id)] = breed
else:
    print("Warning: labels.txt no encontrado en", LABELS_TXT)

def load_gt_data_from_yolo_multi_breed(image_dir, label_dir, id_to_breed_map):
    gt_data = []
    if not os.path.exists(label_dir):
        print("Warning: label_dir no existe:", label_dir)
        return gt_data

    for label_filename in os.listdir(label_dir):
        if not label_filename.endswith('.txt'): continue

        base = label_filename.replace(".txt", "")
        img_path = os.path.join(image_dir, base + ".jpg")
        label_path = os.path.join(label_dir, label_filename)

        try:
            w, h = Image.open(img_path).size
        except:
            continue

        boxes = []
        for line in open(label_path):
            cls, xc, yc, ww, hh = map(float, line.split())
            cls = int(cls)
            breed = id_to_breed_map.get(cls)
            if breed is None: continue
            xc *= w; yc *= h; ww *= w; hh *= h
            x1 = int(xc - ww/2)
            y1 = int(yc - hh/2)
            x2 = int(xc + ww/2)
            y2 = int(yc + hh/2)
            boxes.append([x1, y1, x2, y2, breed])

        if boxes:
            gt_data.append({"path": img_path, "gt_boxes": boxes})

    return gt_data


gt_data = load_gt_data_from_yolo_multi_breed(TEST_IMAGES_DIR, LABELS_DIR, ID_TO_BREED)
TOTAL_GT_INSTANCES = sum(len(x["gt_boxes"]) for x in gt_data)
print(f"GT cargado: {len(gt_data)} imágenes, {TOTAL_GT_INSTANCES} instancias.")



# EVALUACIÓN COMPLETA
def evaluate_pipeline(gt_data, yolo_mode='ULTRALYTICS', yolo_onnx_session=None, resnet_variant='FP32'):
    detection_results = []
    classification_results = []
    total_det_lat = 0.0
    total_class_lat = 0.0

    for idx, item in enumerate(gt_data):
        img = Image.open(item["path"]).convert("RGB")
        arr = np.array(img)

        # DETECCIÓN
        dets = []
        det_lat = 0.0
        if yolo_mode == 'ULTRALYTICS' and YOLO_MODEL_ORIGINAL is not None:
            dets, det_lat = detect_with_ultralytics(arr, conf_threshold=0.25, yolo_model=YOLO_MODEL_ORIGINAL)
        elif yolo_mode == 'ONNX' and yolo_onnx_session is not None:
            dets, det_lat = detect_with_onnx(arr, yolo_onnx_session, input_size=640, conf_thr=0.25)
        else:
            dets = []
            det_lat = 0.0

        total_det_lat += det_lat

        for gt in item["gt_boxes"]:
            x1, y1, x2, y2, gt_b = gt
            crop = img.crop((x1, y1, x2, y2))

            pred, _, latc = classify_dog_breed(crop, model_variant=resnet_variant)
            total_class_lat += latc

            classification_results.append({
                "gt_breed": gt_b,
                "predicted_breed": pred
            })

        # Guardamos detecciones
        for d in dets:
            detection_results.append({
                "image_id": idx,
                "class_name": d["class"],
                "conf": d["conf"],
                "x1": d["box"][0], "y1": d["box"][1],
                "x2": d["box"][2], "y2": d["box"][3]
            })

    df_det = pd.DataFrame(detection_results)
    return df_det, classification_results, total_det_lat, total_class_lat


# MÉTRICAS
def calculate_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def get_detection_metrics(df, gt_data, total_gt, thr=0.5):
    tp = fp = 0
    matched = set()
    preds = df[df["conf"] >= thr].sort_values("conf", ascending=False)

    for _, pr in preds.iterrows():
        pbox = [pr["x1"], pr["y1"], pr["x2"], pr["y2"]]
        img_id = pr["image_id"]

        best_iou = 0
        best_idx = -1

        for j, gt in enumerate(gt_data[img_id]["gt_boxes"]):
            if (img_id, j) in matched: continue
            iou = calculate_iou(pbox, gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= thr:
            tp += 1
            matched.add((img_id, best_idx))
        else:
            fp += 1

    fn = total_gt - len(matched)
    P = tp/(tp+fp+1e-6)
    R = tp/(tp+fn+1e-6)
    F1 = 2*P*R/(P+R+1e-6)
    return {"Precision": P, "Recall": R, "F1-score": F1}

def get_classification_accuracy(res_list):
    correct = sum(1 for x in res_list if x["gt_breed"] == x["predicted_breed"])
    total = len(res_list)
    return (correct/total if total else 0), total


# MAIN
def main():
    # Cuantizar ResNet
    print("\n--- Cuantizando ResNet18 (dinámico, INT8) ---")
    try:
        model_int8 = quantize_resnet_dynamic(save_path="resnet18_int8.pth")
        print("OK: ResNet18 cuantizado (dynamic).")
    except Exception as e:
        print("Fallo cuantización ResNet:", e)
        model_int8 = None

    # Intentar cuantizar YOLO a ONNX INT8
    yolo_onnx_int8_path = None
    if HAS_ULTRALYTICS:
        print("\n--- Intentando exportar + cuantizar YOLOv8 -> ONNX INT8 ---")
        try:
            yolo_onnx_int8_path = try_quantize_yolo_to_onnx_int8(YOLO_MODEL_ORIGINAL)
            if yolo_onnx_int8_path:
                print("YOLO ONNX INT8 listo en:", yolo_onnx_int8_path)
            else:
                print("NO se generó modelo ONNX INT8.")
        except Exception as e:
            print("Error en flujo YOLO->ONNX->INT8:", e)
            yolo_onnx_int8_path = None
    else:
        print("Ultralytics no disponible: no se cuantizará YOLO.")

    # Si existe ONNX INT8, creamos sesión de inference con onnxruntime
    onnx_session = None
    if yolo_onnx_int8_path and HAS_ONNXRUNTIME:
        try:
            print("Cargando sesión ONNXRuntime para modelo cuantizado...")
            sess_options = ort.SessionOptions()
            # si tienes GPU/DirectML/TensorRT provider, puedes añadir providers aquí
            onnx_session = ort.InferenceSession(yolo_onnx_int8_path, sess_options, providers=['CPUExecutionProvider'])
            print("ONNXSession creada.")
        except Exception as e:
            print("No se pudo crear onnxruntime session:", e)
            onnx_session = None

    # Evaluar pipeline original
    print("\n--- Ejecutando pipeline original (Ultralytics YOLO + ResNet FP32) ---")
    df_det_orig, cls_orig, det_lat_o, cls_lat_o = evaluate_pipeline(gt_data, yolo_mode='ULTRALYTICS', yolo_onnx_session=None, resnet_variant='FP32')

    # Evaluar pipeline con ResNet INT8
    print("\n--- Ejecutando pipeline con ResNet INT8 (detección: Ultralytics original) ---")
    df_det_rint8, cls_rint8, det_lat_rint8, cls_lat_rint8 = evaluate_pipeline(gt_data, yolo_mode='ULTRALYTICS', yolo_onnx_session=None, resnet_variant='INT8')

    # Si tenemos ONNX INT8 para YOLO, evaluar detección con ese modelo
    df_det_yolo_onnx = None
    cls_yolo_onnx = None
    det_lat_yolonnx = cls_lat_yolonnx = 0.0
    if onnx_session is not None:
        print("\n--- Ejecutando pipeline con YOLO ONNX INT8 (si disponible) + ResNet INT8 ---")
        df_det_yolo_onnx, cls_yolo_onnx, det_lat_yolonnx, cls_lat_yolonnx = evaluate_pipeline(gt_data, yolo_mode='ONNX', yolo_onnx_session=onnx_session, resnet_variant='INT8')

    # 7) Calculamos métricas
    print("\n===== RESULTADOS =====\n")
    # Detección: original
    m_det_o = get_detection_metrics(df_det_orig, gt_data, TOTAL_GT_INSTANCES)
    print("Detección YOLO (Original ultralytics):", m_det_o)
    print(f"Latencia detección (avg por imagen): {det_lat_o / max(1, len(gt_data)):.2f} ms/img")

    # Detección: quantized ONNX
    if df_det_yolo_onnx is not None:
        m_det_q = get_detection_metrics(df_det_yolo_onnx, gt_data, TOTAL_GT_INSTANCES)
        print("Detección YOLO (ONNX INT8):", m_det_q)
        print(f"Latencia detección (ONNX INT8 avg): {det_lat_yolonnx / max(1, len(gt_data)):.2f} ms/img")
    else:
        print("Detección YOLO ONNX INT8: no disponible (fallback a original).")

    # Clasificación: ResNet FP32 vs INT8
    acc_o, tot = get_classification_accuracy(cls_orig)
    acc_int8, _ = get_classification_accuracy(cls_rint8)
    print("\nClasificación ResNet:")
    print(f"Accuracy Original (FP32): {acc_o:.4f} (n={tot})")
    print(f"Accuracy Cuantizado (INT8 dyn): {acc_int8:.4f} (n={tot})")
    print(f"Latencia ResNet FP32 (avg por crop): {cls_lat_o / max(1, tot):.2f} ms/crop")
    print(f"Latencia ResNet INT8 (avg por crop): {cls_lat_rint8 / max(1, tot):.2f} ms/crop")
main()