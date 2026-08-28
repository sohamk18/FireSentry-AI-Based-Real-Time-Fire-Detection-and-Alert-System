FireSentry — AI-Powered Surveillance Camera Fire Detection

FireSentry is a real-time fire detection system that turns a standard webcam or surveillance camera into an intelligent early-warning tool. It combines a fine-tuned MobileNetV2 CNN with a lightweight HSV color-mask heuristic to flag fire in live video, then automatically alerts people via SMS and WhatsApp with the detected location.

Unlike simple single-frame classifiers, the system reduces false positives by requiring:

Model confidence above a probability threshold, averaged over a rolling window of recent frames
Color-based confirmation — a minimum fraction of fire-colored (red/orange/yellow) pixels via HSV masking
Temporal consistency — several consecutive confirmed frames before triggering an alert
How it works
Captures live video via OpenCV (fire_AI.py)
Preprocesses each frame (resize to 224×224, normalize) for the CNN
Runs inference through a MobileNetV2-based binary classifier (fire / no fire)
Applies a moving-average probability filter + HSV color-mask check to suppress false alarms
On confirmed detection:
Plays a local audio alarm
Fetches approximate location via IP geolocation + reverse geocoding (Geopy)
Sends SMS and WhatsApp alerts through the Twilio API
Model training
Trained on the D-Fire dataset (YOLO-format annotations) using a custom YOLODataset loader that converts bounding-box labels into binary fire/no-fire classification labels
Transfer learning on MobileNetV2 (ImageNet weights, frozen base) with a custom dense classification head
Evaluated separately on a held-out test split for accuracy

*Tech stack*
Python, TensorFlow / Keras (MobileNetV2 CNN)
OpenCV — video capture, frame processing, HSV color masking
NumPy
Twilio API — SMS & WhatsApp alerts
Geopy — reverse geocoding for location in alerts
playsound — local audio alarm

Repo contents

File	Purpose
fire_AI.py	Main real-time detection loop + alert system
detector.py	Model training script (MobileNetV2 + D-Fire dataset)
yolo_data_loader.py	Converts YOLO-format labels into binary classification labels
accuracy detect.py	Evaluates trained model accuracy on the test set
inspect_model.py / inspect_h5.py	Utilities to inspect the saved .h5 model architecture/weights
run_detection_test.py	Runs detection on a single sample image for debugging
fire_detection_model1.h5	Trained model weights
Alarm.mp3	Alarm sound played on detection
