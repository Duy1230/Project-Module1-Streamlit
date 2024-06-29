import cv2
import numpy as np
from PIL import Image
import gradio as gr

MODEL = "../assets/models/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "../assets/models/MobileNetSSD_deploy.prototxt.txt"


def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


def annotate_image(image, detections, confidence_threshold=0.5):
    # loop over the detections
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            # extract the index of the class label from the 'detections',
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(image, (start_x, start_y),
                          (end_x, end_y), (0, 255, 0), 2)
    return image


def object_detection(input_image):
    # Convert to numpy array if it's not already
    if isinstance(input_image, Image.Image):
        image = np.array(input_image)
    else:
        image = input_image

    # Ensure the image is in RGB format
    if image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    detections = process_image(image)
    processed_image = annotate_image(image, detections)

    return processed_image


iface = gr.Interface(
    fn=object_detection,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(),
    title="Object Detection for Images",
    description="Upload an image to detect objects."
)

if __name__ == "__main__":
    iface.launch()
