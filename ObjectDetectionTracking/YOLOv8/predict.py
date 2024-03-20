from ultralytics import YOLO


model = YOLO('best_yolo8x.pt')

results = model(source="datasets/cell_no_classes_detection/test/images/Timepoint_001_220518-ST_C03_s3.jpg", 
                conf=0.5, save=True, imgsz=2048, save_txt=False, save_crop=True, show_labels=False, show_conf=True)