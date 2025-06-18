from ultralytics import YOLO

model = YOLO("yolo_screw_model/weights/best.pt")  # Load the trained model
metrics = model.val()  # Validate the model on the validation dataset
print(metrics)  # Print the evaluation metrics