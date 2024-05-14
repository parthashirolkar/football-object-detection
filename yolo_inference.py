from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

results = model.predict("input_videos/0b1495d3_1.mp4", save=True, stream=True)