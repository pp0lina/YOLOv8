from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Run inference with segmentation and object detection
results = model.predict(source="testData/test1.mov", save=True, stream=True)


for r in results:
    boxes = r.boxes 
    masks = r.masks
    probs = r.probs

    # Retrieve inference time from the speed attribute
    inference_time = r.speed.get('inference', None) if r.speed else None

