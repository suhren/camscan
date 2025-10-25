import cv2
import customtkinter as ctk

from app.utils import opencv_to_ctk_image


class ImageViewerWindow(ctk.CTkToplevel):
    def __init__(
        self,
        master: ctk.CTkToplevel,
        name: str,
        image: cv2.Mat,
        window_width: int,
        window_height: int,
    ):
        super().__init__(master=master)

        self.name = name
        self.image = image
        self.window_width = window_width
        self.window_height = window_height

        self._current_size = [0, 0]

        self._window = ctk.CTkToplevel()
        self._frame_widget = ctk.CTkFrame(master=self._window)
        self._image_widget = ctk.CTkLabel(master=self._frame_widget, text=None)

        self._window.geometry(f"{self.window_width}x{self.window_height}")
        self._window.title(self.name)

        # Bind an event to when the window changes size to resize the image
        self._window.bind("<Configure>", self._on_window_resize)

        # Pack the widgets
        self._frame_widget.pack(fill=ctk.BOTH, expand=True)
        self._image_widget.pack()

    def _on_window_resize(self, event):
        # We need to make sure that the only widget that is allowed to
        # trigger the image resizing is the window itself. Otherwise, when
        # we update the image size, the image widget itself will generate
        # a 'Configure' event which will trigger this function again. That
        # will lead to an endless stream of events.
        if event.widget != self._window:
            return
        # We are only interested in updating the image size if the window
        # changes size. This event is also triggered when the window moves,
        # so we keep track of the current window size and compare to the new
        # one and update only if it changes.
        new_size = [event.width, event.height]
        if self._current_size == new_size:
            return
        # Modify values inplace to keep the reference intact
        self._current_size[:] = new_size[:]
        # Make sure that the containing frame widget has been updated
        # since it is that size which determines the maximum size of
        # the displayed image inside. In some cases, like when the user
        # expands the window to full-screen, this frame might not have
        # enough time to update before we attempt to update time image
        # within. Then it is not possible to fully expand the image to
        # the entire frame size. By manually calling the update here we
        # can ensure that the frame is the maximum size first.
        self._frame_widget.update()
        self._resize_image()

    def _resize_image(self):
        """Resize the image to fill up the frame in the window"""
        max_width = self._frame_widget.winfo_width()
        max_height = self._frame_widget.winfo_height()

        # At startup, this area might be of size zero. If so, try later
        if not (max_width > 1 and max_height > 1):
            return

        # Convert the OpenCV image to a CTkImage to display in the widget
        new_image = opencv_to_ctk_image(
            image=self.image, width=max_width, height=max_height
        )
        self._image_widget.photo = new_image
        self._image_widget.configure(image=new_image)

    def show(self):
        """
        Open an image viewer window displaying the current image of this Entry.
        """
        # The current window size. Used to keep track of when it changes
        current_size = [0, 0]

        # Make sure this window is on top of the main window
        self._window.lift()
        self._window.attributes("-topmost", True)
        # It can take some time for the window to set up its widgets and get its
        # proper size. Therefore, wait a little bit before displaying the image.
        self._window.after(ms=100, func=self._resize_image)
