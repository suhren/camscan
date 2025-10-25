import cv2
import customtkinter as ctk

from app.utils import opencv_to_ctk_image


class CameraFeed(ctk.CTkFrame):
    def __init__(
        self,
        master: ctk.CTkBaseClass,
    ):
        super().__init__(master=master, fg_color="transparent")
        # frame = ctk.CTkFrame()
        self._camera_image_widget = ctk.CTkLabel(
            self,
            text=None,
            padx=0,
            pady=0,
        )
        self._camera_image_label = ctk.CTkLabel(
            self,
            text="No Camera",
            font=ctk.CTkFont(size=20, weight="bold"),
            padx=0,
            pady=0,
        )
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._camera_image_widget.grid(row=0, column=0, sticky="nsew")
        self._camera_image_label.grid(row=0, column=0, sticky="nsew")
        self._camera_image_widget.lift()

    def display(self, image: cv2.Mat):
        """
        This function is continuously called to show the camera feed.
        """
        # Get the current width and height of the frame area
        max_width = self.winfo_width()
        max_height = self.winfo_height()

        # At startup, this area might still be of size zero. If so, try later
        if not (max_width > 1 and max_height > 1):
            return

        if image is not None:
            # Convert the OpenCV image to a CTkImage to display in the widget
            image_width = image.shape[1]
            image_height = image.shape[0]
            # If the image is larger than the max widget size, resize it first
            if image_width > max_width or image_height > max_height:
                image = opencv_to_ctk_image(
                    image=image, width=max_width, height=max_height
                )
            else:
                image = opencv_to_ctk_image(image=image)
            # Update the camera image widget
            self._camera_image_widget.photo = image
            self._camera_image_widget.configure(image=image)
            # Ensure the camera image widget is top of the 'No Camera' widget
            self._camera_image_widget.lift()
        else:
            # If there was no image captured, lift the 'No Camera' widget on top
            self._camera_image_label.lift()
