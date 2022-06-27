import cv2
import numpy as np


def stack_images(scale, images):
    rows = len(images)
    cols = len(images[0])
    rowsAvailable = isinstance(images[0], list)
    width = images[0][0].shape[1]
    height = images[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if images[x][y].shape[:2] == images[0][0].shape[:2]:
                    images[x][y] = cv2.resize(images[x][y], (0, 0), None, scale, scale)
                else:
                    images[x][y] = cv2.resize(
                        images[x][y],
                        (images[0][0].shape[1], images[0][0].shape[0]),
                        None,
                        scale,
                        scale,
                    )
                if len(images[x][y].shape) == 2:
                    images[x][y] = cv2.cvtColor(images[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(images[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if images[x].shape[:2] == images[0].shape[:2]:
                images[x] = cv2.resize(images[x], (0, 0), None, scale, scale)
            else:
                images[x] = cv2.resize(
                    images[x],
                    (images[0].shape[1], images[0].shape[0]),
                    None,
                    scale,
                    scale,
                )
            if len(images[x].shape) == 2:
                images[x] = cv2.cvtColor(images[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(images)
        ver = hor
    return ver


def new_coordinates_resize(original_size, new_size, original_coordinate):
    original_size = np.array(original_size)
    new_size = np.array(new_size)
    original_coordinate = np.array(original_coordinate)
    xy = original_coordinate / (original_size / new_size)
    x, y = int(xy[0]), int(xy[1])
    x = x if x > 0 else 0
    y = y if y > 0 else 0
    return np.array([x, y], dtype=np.uint16)


def setup_resolution(size_each_camera_image, resize_all_camera_image, total_cam):
    if total_cam % 2 == 0:
        return stack_images(
            resize_all_camera_image,
            (
                [
                    np.random.randint(
                        0, 255, size=(*size_each_camera_image, 3), dtype=np.uint8
                    )
                    for _ in range(0, total_cam // 2)
                ],
                [
                    np.random.randint(
                        0, 255, size=(*size_each_camera_image, 3), dtype=np.uint8
                    )
                    for _ in range(total_cam // 2, total_cam)
                ],
            ),
        ).shape[:2]
    else:
        return stack_images(
            resize_all_camera_image,
            (
                [
                    np.random.randint(
                        0, 255, size=(*size_each_camera_image, 3), dtype=np.uint8
                    )
                    for _ in range(total_cam)
                ],
            ),
        ).shape[:2]
