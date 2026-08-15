"""
Camera viewer widget.
"""

import typing as t

import customtkinter as ctk
import cv2

from camscan import types, utils


class ImagePreview(ctk.CTkFrame):
    def __init__(self, master: t.Any, **kwargs: t.Any) -> None:
        super().__init__(master=master, **kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.image_widget = ctk.CTkLabel(
            master=self,
            text=None,
            padx=0,
            pady=0,
        )
        self.message_label = ctk.CTkLabel(
            master=self,
            text="",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.info_label = ctk.CTkLabel(
            master=self,
            text="",
            font=ctk.CTkFont(size=14, family="monospace"),
        )

        self.image_widget.grid(row=0, column=0, sticky="nsew")
        self.message_label.grid(row=0, column=0, sticky="nsew")
        self.info_label.grid(row=0, column=0, sticky="se")

    def get_width(self) -> int:
        return self.image_widget.winfo_width()

    def get_height(self) -> int:
        return self.image_widget.winfo_height()

    def show(
        self,
        image: types.Image | None = None,
        message: str | None = None,
        info: str | None = None,
    ) -> None:

        # Get the current width and height of the camera widget area
        max_width = self.get_width()
        max_height = self.get_height()

        # At startup, this area might still be of size zero. If so, try later
        if image is not None:  # and (max_width > 1 and max_height > 1):
            # The image must have three color channels, so convert if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            image_width = image.shape[1]
            image_height = image.shape[0]

            # If the image is larger than the max widget size, resize it first
            if (image_width > max_width or image_height > max_height) and (
                max_width > 1 and max_height > 1
            ):
                image = utils.opencv_to_ctk_image(
                    image=image, width=max_width, height=max_height
                )
            else:
                image = utils.opencv_to_ctk_image(image=image)

            self.image_widget.configure(image=image)
            self.image_widget.grid()
        else:
            self.image_widget.grid_remove()

        if message is not None:
            self.message_label.configure(text=message)
            self.message_label.grid()
        else:
            self.message_label.configure(text="")
            self.message_label.grid_remove()

        if info is not None:
            self.info_label.configure(text=info)
            self.info_label.grid()
        else:
            self.info_label.configure(text="")
            self.info_label.grid_remove()
