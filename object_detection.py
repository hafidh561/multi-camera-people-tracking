import cv2
import numpy as np
import onnxruntime as ort


class ObjectDetection:
    def __init__(
        self,
        onnx_path="./pretrained_models/yolov4-tiny.onnx",
        coco_names_path="./pretrained_models/coco.names",
        device="cpu",
        confidence_threshold=0.5,
        choose_classes=["person"],
    ):
        self.onnx_path = onnx_path
        self.coco_names_path = coco_names_path
        self.confidence_threshold = confidence_threshold
        self.choose_classes = np.array(choose_classes)
        self.device = device
        self.nms_threshold = (
            0
            if self.confidence_threshold - 0.1 < 0
            else self.confidence_threshold - 0.1
        )

        with open(self.coco_names_path, "r") as f:
            self.class_names = f.readlines()
            self.class_names = np.array(
                [cls.replace("\n", "") for cls in self.class_names]
            )

        if self.device.lower() == "cpu":
            if ort.get_device() == "CUDA":
                print("CUDA available, if you want to switch your device into CUDA")
            self.ort_session = ort.InferenceSession(
                self.onnx_path, providers=["CPUExecutionProvider"]
            )
        elif self.device.lower() == "cuda":
            self.ort_session = ort.InferenceSession(
                self.onnx_path,
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            )
        else:
            raise ValueError("Choose between CPU or CUDA!")

        self.model_height, self.model_width = self.ort_session.get_inputs()[0].shape[
            2:4
        ]

    def predict_img(self, img):
        image = self._preprocessing_img(img)
        input_onnx = self.ort_session.get_inputs()[0].name
        output_onnx = self.ort_session.run(None, {input_onnx: image})
        postprocess_onnx = self._postprocessing_onnx(output_onnx)
        result_outputs = self._postprocessing_result(postprocess_onnx)
        return result_outputs

    def _preprocessing_img(self, img):
        image = cv2.resize(
            img, (self.model_width, self.model_height), interpolation=cv2.INTER_LINEAR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image /= 255.0
        return image

    def _postprocessing_onnx(self, output_onnx):
        box_array = output_onnx[0]
        confs = output_onnx[1]
        num_classes = confs.shape[2]
        box_array = box_array[:, :, 0]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > self.confidence_threshold
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]
            bboxes = []
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = self._nmsbbox(ll_box_array, ll_max_conf, self.nms_threshold)
                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )
            bboxes_batch.append(bboxes)

        return np.array(bboxes_batch, dtype=np.float16)

    def _postprocessing_result(self, postprocess_onnx):
        result_outputs = []
        for x1, y1, x2, y2, _, confidence, label in postprocess_onnx[0]:
            if self.class_names[int(label)] not in self.choose_classes:
                continue
            x1 = int(x1 * self.model_width)
            y1 = int(y1 * self.model_height)
            x2 = int(x2 * self.model_width)
            y2 = int(y2 * self.model_height)
            result_outputs.append(
                {
                    self.class_names[int(label)].title(): {
                        "confidence": float(f"{confidence:.2f}"),
                        "bounding_box": [x1, y1, x2, y2],
                    }
                }
            )

        return np.array(result_outputs)

    def _nmsbbox(self, bbox, max_confidence, min_mode=False):
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = max_confidence.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]
            keep.append(idx_self)
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)
            inds = np.where(over <= self.nms_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.uint8)


if __name__ == "__main__":
    object_detection = ObjectDetection()
    input = cv2.imread("./sample_image/testing.png")
    output = object_detection.predict_img(input)
    print(output)
