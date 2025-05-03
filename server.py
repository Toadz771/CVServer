from ultralytics import YOLO
import socket
import json
import base64
import numpy as np
import cv2
import threading
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_path = "yolo11x-seg.pt"
if not os.path.exists(model_path):
    logger.info("Downloading YOLO model...")
    model_url = "https://drive.google.com/uc?export=download&id=1_FBHoGbMnpyJFbOj2jaU0Ir9u1Nt0Jvu"  # Replace with your link
    urllib.request.urlretrieve(model_url, model_path)
    logger.info("Model downloaded")

# Load YOLO model
model = YOLO("yolo11x-seg.pt")

def handle_client(conn, addr):
    logger.info(f"Connected to client: {addr}")
    buffer = b""
    while True:
        try:
            data = conn.recv(65536)
            if not data:
                logger.info(f"Client {addr} disconnected")
                break
            buffer += data
            if b"\n" in buffer:
                message_end = buffer.index(b"\n")
                message_bytes = buffer[:message_end]
                buffer = buffer[message_end + 1:]
                try:
                    message = json.loads(message_bytes.decode("utf-8"))
                    if "type" in message and message["type"] == "image" and "data" in message:
                        image_data = base64.b64decode(message["data"])
                        np_arr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is None:
                            logger.warning("Failed to decode image")
                            conn.send(json.dumps({"object": "none"}).encode("utf-8") + b"\n")
                            continue
                        results = model(frame)
                        detected = False
                        for result in results:
                            if result.boxes:
                                class_id = int(result.boxes.cls[0])
                                class_name = model.names[class_id].lower()
                                confidence = float(result.boxes.conf[0])
                                if confidence > 0.7:
                                    logger.info(f"Detected: {class_name} ({confidence:.2f})")
                                    conn.send(json.dumps({"object": class_name}).encode("utf-8") + b"\n")
                                    detected = True
                                    break
                        if not detected:
                            conn.send(json.dumps({"object": "none"}).encode("utf-8") + b"\n")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    conn.send(json.dumps({"object": "none"}).encode("utf-8") + b"\n")
        except Exception as e:
            logger.error(f"Client error: {e}")
            break
    conn.close()

def main():
    port = int(os.environ.get("PORT", 12345))  # Use Render's PORT or default to 12345
    godot_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    godot_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    godot_server.bind(("0.0.0.0", port))
    godot_server.listen(5)
    logger.info(f"Server started on port {port}, waiting for connections...")

    while True:
        try:
            conn, addr = godot_server.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()
        except KeyboardInterrupt:
            logger.info("Shutting down server")
            break
        except Exception as e:
            logger.error(f"Server error: {e}")

    godot_server.close()

if __name__ == "__main__":
    main()
