from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

import utils


COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def ensure_model(model_path: Path, image_size: int) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    exported_path = Path(YOLO("yolov8n.pt").export(format="onnx", opset=12, imgsz=[image_size, image_size]))
    if exported_path.resolve() != model_path.resolve():
        shutil.move(str(exported_path), str(model_path))
    return model_path


def run_inference(image_path: Path, model_path: Path, output_path: Path, conf: float, nms: float, image_size: int) -> Path:
    utility = utils.Utils()
    model_path = ensure_model(model_path, image_size)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    net = cv2.dnn.readNetFromONNX(str(model_path))
    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        (image_size, image_size),
        (0, 0, 0),
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)
    output = net.forward()
    result = utility.postprocess_onnx(
        output,
        image,
        COCO_CLASS_NAMES,
        confThreshold=conf,
        nmsThreshold=nms,
        font_size=0.5,
        color=(255, 127, 0),
        text_color=(255, 255, 255),
        input_size=[image_size, image_size],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to save result image to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_image = base_dir / "image2.jpg"
    default_model = base_dir / "model" / "yolov8.onnx"
    default_output = base_dir / "outputs" / "image2_detected.jpg"

    parser = argparse.ArgumentParser(description="Run YOLO ONNX inference on a local image.")
    parser.add_argument("--image", type=Path, default=default_image, help="Path to the input image.")
    parser.add_argument("--model", type=Path, default=default_model, help="Path to the YOLO ONNX model.")
    parser.add_argument("--output", type=Path, default=default_output, help="Path to save the result image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold.")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size for the model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_inference(args.image, args.model, args.output, args.conf, args.nms, args.imgsz)
    print(f"Detection result saved to: {output_path}")


if __name__ == "__main__":
    main()
