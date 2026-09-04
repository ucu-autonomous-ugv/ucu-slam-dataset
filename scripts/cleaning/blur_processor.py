import time as t
from pathlib import Path

import numpy as np
import torch
from gen2.script.detectron2.export.torchscript_patch import patch_instances
from gen2.script.predictor import PATCH_INSTANCES_FIELDS, ClassID, EgoblurDetector
from gen2.script.utils import get_image_tensor, visualize
from rosbags.typesys.store import Typestore

from utils.images import array_to_msg, msg_to_array

RESIZE_MIN: int = 1200
RESIZE_MAX: int = 1200


class BlurProcessor:
    def __init__(
        self,
        face_model_path: Path,
        license_plate_model_path: Path,
        device: str,
        face_threshold: float,
        license_plate_threshold: float,
        nms_iou_threshold: float,
        scale_factor_detections: float,
        typestore: Typestore,
    ):
        self.face_detector = EgoblurDetector(
            model_path=face_model_path,
            device=device,
            detection_class=ClassID.FACE,
            score_threshold=face_threshold,
            nms_iou_threshold=nms_iou_threshold,
            resize_aug={
                "min_size_test": RESIZE_MIN,
                "max_size_test": RESIZE_MAX,
            },
        )
        self.lp_detector = EgoblurDetector(
            model_path=license_plate_model_path,
            device=device,
            detection_class=ClassID.LICENSE_PLATE,
            score_threshold=license_plate_threshold,
            nms_iou_threshold=nms_iou_threshold,
            resize_aug={
                "min_size_test": RESIZE_MIN,
                "max_size_test": RESIZE_MAX,
            },
        )
        self.typestore = typestore
        self.scale_factor_detections = scale_factor_detections

    def blur_batch(self, bgr_arrs: list[np.ndarray]) -> list[np.ndarray]:
        if not bgr_arrs:
            return []

        images = [bgr_arr.copy() for bgr_arr in bgr_arrs]
        image_tensors = [get_image_tensor(bgr_arr.copy()) for bgr_arr in bgr_arrs]
        batched_tensor = torch.stack(image_tensors)
        detections_by_image = [[] for _ in bgr_arrs]

        with patch_instances(fields=PATCH_INSTANCES_FIELDS):
            face_results = self.face_detector.run(batched_tensor)

            if face_results:
                assert len(face_results) == len(bgr_arrs), (
                    "EgoblurDetector.run is expected to return results for "
                    "each image in the batch."
                )

                for idx, detections in enumerate(face_results):
                    detections_by_image[idx].extend(detections)

            lp_results = self.lp_detector.run(batched_tensor)

            if lp_results:
                assert len(lp_results) == len(bgr_arrs), (
                    "EgoblurDetector.run is expected to return results for "
                    "each image in the batch."
                )

                for idx, detections in enumerate(lp_results):
                    detections_by_image[idx].extend(detections)

        inference_time = (
            self.face_detector.last_inference_time
            + self.lp_detector.last_inference_time
        )
        print(f"Inference time: {inference_time:.3f} seconds")

        start_time = t.time()
        blurred_images = [
            visualize(image, detections, self.scale_factor_detections)
            for image, detections in zip(images, detections_by_image)
        ]
        end_time = t.time()
        print(f"Blurring time: {end_time - start_time:.3f} seconds")

        return blurred_images

    def blur(self, bgr_arr: np.ndarray) -> np.ndarray:
        return self.blur_batch([bgr_arr])[0]

    def __call__(self, msg):
        bgr_arr = msg_to_array(msg, is_bgr=True)
        header = msg.header
        encoding = msg.encoding

        bgr_arr_blured = self.blur(bgr_arr)

        msg = array_to_msg(
            bgr_arr_blured,
            encoding=encoding,
            typestore=self.typestore,
            is_bgr=True,
        )
        msg.header = header

        return msg

    def close(self) -> None:
        pass
