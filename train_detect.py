from ultralytics import YOLO

# Load a base YOLOv8 segmentation model
model = YOLO("yolov8n.pt")

#Start training the model
model.train(
    data="data.yaml", #path to the dataset configuration file
    epochs=80, #number of training epochs
    imgsz=640, #input image size
    batch=3, #adjust batch size according to your GPU memory
    name="yolo_screw_model", #name of the training run
    project=".", #project directory
    exist_ok=True, #overwrite existing project
    save_period=0, #save model every epoch
    verbose=False, #disable verbose output
    augment=True, #enable data augmentation
 )