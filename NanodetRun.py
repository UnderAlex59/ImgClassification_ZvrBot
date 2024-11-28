import numpy as np
import cv2 as cv
from Nanodet import NanoDet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU],
]

classes = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()

    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(
                img,
                0,
                0,
                left,
                target_size[1] - neww - left,
                cv.BORDER_CONSTANT,
                value=0,
            )  # add border
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(
                img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0
            )
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)

    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale


def unletterbox(bbox, original_image_shape, letterbox_scale):
    ret = bbox.copy()

    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale

    if h == w:
        ratio = h / newh
        ret = ret * ratio
        return ret

    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)

    return ret.astype(np.int32)


def vis(preds, res_img, letterbox_scale, fps=None):
    ret = res_img.copy()

    # draw FPS
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(ret, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # draw bboxes and labels
    for pred in preds:
        bbox = pred[:4]
        conf = pred[-2]
        classid = pred[-1].astype(np.int32)

        # bbox
        xmin, ymin, xmax, ymax = unletterbox(bbox, ret.shape[:2], letterbox_scale)
        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

        # label
        label = "{:s}: {:.2f}".format(classes[classid], conf)
        cv.putText(
            ret,
            label,
            (xmin, ymin - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            thickness=2,
        )

    return ret


def classify_nanodet(img_path: str):
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]
    print(target_id)

    model = NanoDet(
        modelPath="object_detection_nanodet_2022nov.onnx",
        prob_threshold=0.35,
        iou_threshold=0.6,
        backend_id=backend_id,
        target_id=target_id,
    )
    """Choose one of the backend-target pair to run this demo:
                            {:d}: (default) OpenCV implementation + CPU,
                            {:d}: CUDA + GPU (CUDA),
                            {:d}: CUDA + GPU (CUDA FP16),
                            {:d}: TIM-VX + NPU,
                            {:d}: CANN + NPU
                        """

    tm = cv.TickMeter()
    tm.reset()

    image = cv.imread(img_path)
    image = cv.copyMakeBorder(
        image,
        top=50,
        bottom=50,
        left=50,
        right=50,
        borderType=cv.BORDER_CONSTANT,
        value=0,
    )
    input_blob = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Letterbox transformation
    input_blob, letterbox_scale = letterbox(input_blob)

    # Inference
    tm.start()
    preds = model.infer(input_blob)
    tm.stop()

    img = vis(preds, image, letterbox_scale)

    res_name = img_path.split(".")[0] + "_NanoDet." + img_path.split(".")[1]
    cv.imwrite(res_name, img)
    return format("Время классификации: {:.2f} мс".format(tm.getTimeMilli()))
