from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import numpy as np
import cv2
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = YOLO("yolo11n-seg.pt")

@app.route("/detect", methods=["POST"])
def detect_object():
    try:
        data = request.get_json()
        if not data or "type" not in data or data["type"] != "image" or "data" not in data:
            return jsonify({"object": "none"}), 400
        image_data = base64.b64decode(data["data"])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Failed to decode image")
            return jsonify({"object": "none"}), 400
        results = model(frame)
        for result in results:
            if result.boxes:
                class_id = int(result.boxes.cls[0])
                class_name = model.names[class_id].lower()
                confidence = float(result.boxes.conf[0])
                if confidence > 0.7:
                    logger.info(f"Detected: {class_name} ({confidence:.2f})")
                    return jsonify({"object": class_name})
        return jsonify({"object": "none"})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"object": "none"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)