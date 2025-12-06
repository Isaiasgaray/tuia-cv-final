from ultralytics import YOLO
import os

try:
    YOLO_MODEL = YOLO('yolov8n.pt')
    for k, v in YOLO_MODEL.names.items():
        if v == 'dog':
            CLASS_ID = k
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
    """
    if YOLO_MODEL is None:
        print("El modelo YOLO no está disponible.")
        return []
    
    if not os.path.exists(source):
        print(f"Error: La imagen no se encontró en la ruta: {source}")
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
        boxes = r.boxes
        
        # Filtramos las detecciones para quedarnos solo con la clase 'dog'
        dog_detections = boxes[boxes.cls == CLASS_ID]
        
        for det in dog_detections:
            # Obtenemos las coordenadas y nivel de confianza
            x1, y1, x2, y2 = [int(val) for val in det.xyxy[0]]
            conf = det.conf[0].item()
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': round(conf, 4)
            })

    return detections