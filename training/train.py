from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="training/data.yaml",
        epochs=50,
            imgsz=640,
                batch=8,
                    name="cricket_ball"
                    )