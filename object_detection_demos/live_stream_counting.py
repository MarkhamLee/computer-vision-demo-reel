from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO('yolov8m.pt')
results = model.track(source="https://youtu.be/LNwODJXcvt4",
                      conf=0.3, iou=0.5, show=True)
                for class_id in self.count_object.in_counts:
                    print(f"In count for {self.model.names[0]}: {self.count_object.in_counts[0]}")
                for class_id in self.count_object.out_count:
                    print(f"Out count for {self.model.names[0]}: {self.count_object.out_counts[0]}")
                count = 0