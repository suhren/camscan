import functools
import typing as t

import cv2
import tkinter as tk
import customtkinter as ctk

import app.config as cfg
from app.utils import opencv_to_ctk_image
from app.widgets.image_viewer import ImageViewerWindow


class CaptureEntry(ctk.CTkFrame):
    """
    Helper class for keeping track of the captured images. This class both
    contains the original OpenCV image, as well as the GUI element called an
    'Entry' which is comprised of several underlying widgets.
    :param image: The original OpenCV image capture from the camera
    :param name: A name given to the image which is displayed in the Entry
    :param index: The index number shown in the Entry
    :param master: The parent widget containing the Entry
    :param move_entry: A function used to move this entry up or down in the list
    """

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        image: cv2.Mat,
        name: str,
        index: int,
        move_entry: t.Callable,
    ):
        super().__init__(master=master)

        self.var_selected = tk.IntVar(value=0)
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.name = name
        self.index = index

        self.move_up_button = ctk.CTkButton(
            master=self,
            text="🔼",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=functools.partial(move_entry, self, -1),
        )
        self.selection_checkbox = ctk.CTkCheckBox(
            master=self,
            text=None,
            checkbox_width=24,
            checkbox_height=24,
            width=24,
            height=24,
            variable=self.var_selected,
        )
        self.move_down_button = ctk.CTkButton(
            master=self,
            text="🔽",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=functools.partial(move_entry, self, 1),
        )
        self.image_widget = ctk.CTkButton(
            master=self,
            fg_color="transparent",
            text=None,
            command=self.open_image_viewer_window,
        )
        self.index_label = ctk.CTkLabel(
            master=self,
            text=str(self.index),
        )
        self.name_label = ctk.CTkLabel(
            master=self,
            text=self.name,
        )
        self.index_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="nsew")
        self.name_label.grid(row=0, column=1, padx=5, pady=(5, 0), sticky="nsew")
        self.move_up_button.grid(row=1, column=0, padx=5, pady=5)
        # The checkbox needs slightly more padding on the left to be aligned
        # with the up/down buttons in the same column
        self.selection_checkbox.grid(row=2, column=0, padx=(12, 5), pady=5)
        self.move_down_button.grid(row=3, column=0, padx=5, pady=5)
        self.image_widget.grid(
            row=1, column=1, rowspan=3, padx=5, pady=(0, 5), sticky="nsew"
        )

        self.set_current_image(image=image)

    def set_current_image(self, image: cv2.Mat):
        """
        Update the current displayed image of this Entry. This will not modify
        the original OpenCV image stored in this object. This will also update
        the smaller thumbnail picture used in the Entry.
        :param image: The new OpenCV image to set the current displayed image to
        """
        self.current_image = image.copy()
        thumbnail_image = opencv_to_ctk_image(image=image, width=230, height=400)
        self.image_widget.photo = thumbnail_image
        self.image_widget.configure(image=thumbnail_image)

    def open_image_viewer_window(self):
        """
        Open an image viewer window displaying the current image of this Entry.
        """
        image_viewer_window = ImageViewerWindow(
            master=self,
            name=self.name,
            image=self.current_image,
            window_width=cfg.WINDOW_WIDTH,
            window_height=cfg.WINDOW_HEIGHT,
        )

        image_viewer_window.show()
