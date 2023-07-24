import math

import numpy as np
import cv2


def draw_contour(
    image: cv2.Mat,
    contour: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 4,
) -> cv2.Mat:
    return cv2.polylines(
        img=image,
        pts=[contour],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]

    # No resizing needed
    if width is None and height is None:
        return image

    # Resize to the smallest of both width and height
    if width is not None and height is not None:
        rh = height / float(h)
        rw = width / float(w)
        if rh < rw:
            dim = (int(w * rh), int(height))
        else:
            dim = (int(width), int(h * rw))

    elif width is None:
        r = height / float(h)
        dim = (int(w * r), int(height))
    else:
        r = width / float(w)
        dim = (int(width), int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def images_in_grid(
    images: list | tuple | dict,
    output_width: int,
    output_height: int,
    draw_grid: bool = True,
    grid_color: tuple = (255, 255, 255),
):
    num_images = len(images)

    if num_images == 1:
        num_cols, num_rows = 1, 1
    elif num_images == 2:
        num_cols, num_rows = 2, 1
    elif num_images <= 4:
        num_cols, num_rows = 2, 2
    elif num_images <= 6:
        num_cols, num_rows = 3, 2
    elif num_images <= 9:
        num_cols, num_rows = 3, 3
    elif num_images <= 12:
        num_cols, num_rows = 3, 4
    else:
        size = math.ceil(math.sqrt(num_images))
        num_cols, num_rows = size, size

    subframe_width = int(output_width / num_cols)
    subframe_height = int(output_height / num_rows)

    output_image = np.zeros((output_height, output_width, 3), np.uint8)

    image_labels = None

    if isinstance(images, dict):
        image_labels = list(images.keys())
        images = list(images.values())

    for i, img in enumerate(images):
        col = i % num_cols
        row = i // num_cols
        top_left_x = col * subframe_width
        top_left_y = row * subframe_height

        if img is not None:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            scaled_img = resize_with_aspect_ratio(
                img, width=subframe_width, height=subframe_height
            )

            vertical_padding = (subframe_height - scaled_img.shape[0]) // 2
            horizontal_padding = (subframe_width - scaled_img.shape[1]) // 2

            subframe_top_left_x = top_left_x + horizontal_padding
            subframe_top_left_y = top_left_y + vertical_padding
            subframe_bottom_right_x = subframe_top_left_x + scaled_img.shape[1]
            subframe_bottom_right_y = subframe_top_left_y + scaled_img.shape[0]

            output_image[
                subframe_top_left_y:subframe_bottom_right_y,
                subframe_top_left_x:subframe_bottom_right_x,
                :,
            ] = scaled_img

        if image_labels:
            (label_width, label_height), baseline = cv2.getTextSize(
                text=image_labels[i],
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                thickness=2,
            )
            cv2.putText(
                img=output_image,
                text=image_labels[i],
                org=(top_left_x + 10, top_left_y + label_height + baseline),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    if draw_grid:
        for col in range(1, num_cols):
            cv2.line(
                output_image,
                (col * subframe_width, 0),
                (col * subframe_width, output_height),
                grid_color,
                thickness=2,
                lineType=8,
            )

        for row in range(1, num_rows):
            cv2.line(
                output_image,
                (0, row * subframe_height),
                (output_width, row * subframe_height),
                grid_color,
                thickness=2,
                lineType=8,
            )

    return output_image
