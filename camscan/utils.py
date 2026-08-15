import datetime
import math

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image as PIL_Image

from camscan import types


def get_timestamp_str() -> str:
    """
    Return the current time as a timestamp string.
    :return: A timestamp string.
    """
    return datetime.datetime.now(tz=datetime.UTC).strftime(r"%Y%m%d_%H%M%S_%f")


def opencv_to_pil_image(
    image: types.Image,
    width: int | None = None,
    height: int | None = None,
) -> PIL_Image.Image:
    """
    Given an OpenCV image, convert to to a PIL image. The function also supports
    resizing the image while keeping its original aspect ratio.
    :param image: The input OpenCV image
    :param width: Optional width to scale the image to
    :param height: Optional height to scale the image to
    :raises ValueError: If the input image has a non-compatible shape
    :return: The image converted to a PIL image
    """

    # Check if the image is a black-and-whie image (only 2 dimensions)
    # In this case, it is only a binary 0 for black and 1 for white
    if len(image.shape) == 2:
        pass
    # Check if the image has 3 color dimensions (BGR).
    # If it does, we need to convert from OpenCV BGR to RGB before showing.
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Check if the image has 4 color dimensions (BGRA).
    # If it does, we need to convert from OpenCV BGRA to RGBA before showing.
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unknown image shape: {image.shape}")

    return PIL_Image.fromarray(
        resize_with_aspect_ratio(
            image=image,
            width=width,
            height=height,
        )
    )


def opencv_to_ctk_image(
    image: types.Image,
    width: int | None = None,
    height: int | None = None,
) -> ctk.CTkImage:
    """
    Given an OpenCV image, convert to to a CTkImage. The function also supports
    resizing the image while keeping its original aspect ratio.
    :param image: The input OpenCV image
    :param width: Optional width to scale the image to
    :param height: Optional height to scale the image to
    :return: The image converted to a CTkImage
    """
    pil_image = opencv_to_pil_image(image=image, width=width, height=height)
    return ctk.CTkImage(
        pil_image,
        size=(pil_image.width, pil_image.height),
    )


def draw_contour(
    image: types.Image,
    contour: types.Contour,
    color: types.Color = (0, 255, 0),
    thickness: int = 4,
) -> types.Image:
    return cv2.polylines(
        img=image.copy(),
        pts=[contour],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


def resize_with_aspect_ratio(
    image: types.Image,
    width: float | None = None,
    height: float | None = None,
    inter: int = cv2.INTER_AREA,
) -> types.Image:
    # No resizing needed
    if width is None and height is None:
        return image

    h, w = image.shape[:2]

    # Resize to the smallest of both width and height
    if width is not None and height is not None:
        rh = height / float(h)
        rw = width / float(w)
        if rh < rw:
            dim = (int(w * rh), int(height))
        else:
            dim = (int(width), int(h * rw))

    elif height is not None:
        r = height / float(h)
        dim = (int(w * r), int(height))

    elif width is not None:
        r = width / float(w)
        dim = (int(width), int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def images_in_grid(
    images: list[types.Image | None],
    output_width: int,
    output_height: int,
    draw_grid: bool = True,
    labels: list[str] | None = None,
    grid_color: tuple = (255, 255, 255),
    fontScale: float = 0.5,
) -> types.Image:

    num_images = len(images)

    if labels is not None and len(labels) != len(images):
        raise ValueError("The number of labels must match the number of images")

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

    images_to_show: list[types.Image | None] = []

    # Check if any image has an alpha channel
    if any(
        img is not None and len(img.shape) > 2 and img.shape[2] == 4 for img in images
    ):
        output_image = np.zeros((output_height, output_width, 4), np.uint8)
        for img in images:
            if img is None:
                images_to_show.append(None)
            elif len(img.shape) == 2:
                images_to_show.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA))
            elif img.shape[2] == 3:
                images_to_show.append(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
            else:
                images_to_show.append(img)
    else:
        output_image = np.zeros((output_height, output_width, 3), np.uint8)
        for img in images:
            if img is None:
                images_to_show.append(None)
            elif len(img.shape) == 2:
                images_to_show.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            elif img.shape[2] == 4:
                images_to_show.append(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
            else:
                images_to_show.append(img)

    for i, img in enumerate(images_to_show):
        col = i % num_cols
        row = i // num_cols
        top_left_x = col * subframe_width
        top_left_y = row * subframe_height

        if img is not None:
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

        if labels is not None:
            text = f"{i + 1}: {labels[i]}"

            (_label_width, label_height), _baseline = cv2.getTextSize(
                text=text,
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=fontScale,
                thickness=1,
            )
            label_x0 = top_left_x + 10
            label_x1 = label_x0 + _label_width
            label_y0 = top_left_y + 10
            label_y1 = label_y0 + label_height
            cv2.rectangle(
                img=output_image,
                pt1=(label_x0 - 5, label_y0 - 5),
                pt2=(label_x1 + 5, label_y1 + 5),
                color=(0, 0, 0),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=output_image,
                text=text,
                org=(label_x0, label_y1),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=fontScale,
                color=(255, 255, 255),
                thickness=1,
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
