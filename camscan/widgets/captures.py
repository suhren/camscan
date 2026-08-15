"""
Widgets related to image captures.
"""

import functools
import tkinter as tk
import typing as t

import customtkinter as ctk

from camscan import config, types, utils
from camscan.logging import logger


class ImageViewer(ctk.CTkToplevel):
    """
    Image viewer window displaying an image.
    """

    def __init__(self, image: types.Image, name: str = "Image") -> None:
        super().__init__()

        self.name = name
        self.image = image
        self.title(name)
        self.closed = False

        # The current window size. Used to keep track of when it changes
        self.current_size = [0, 0]

        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.frame_widget = ctk.CTkFrame(master=self)
        self.image_widget = ctk.CTkLabel(master=self.frame_widget, text=None)

        # Pack the widgets
        self.frame_widget.pack(fill=ctk.BOTH, expand=True)
        self.image_widget.pack()
        # Bind event when window is closes
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)
        # Bind an event to when the window changes size to resize the image
        self.bind("<Configure>", self._on_window_resize)
        # Make sure this window is on top of the main window
        self.lift()
        self.attributes("-topmost", True)
        # It can take some time for the window to set up its widgets and get its
        # proper size. Therefore, wait a little bit before displaying the image.
        self.after(ms=100, func=self._resize_image)

    def _resize_image(self) -> None:
        """
        Resize the image to fill up the frame in the window

        NOTE: If we have just closed the window, some underlying widgets might have
        become unavailable and will cause an error like
        _tkinter.TclError: bad window path name ".!imageviewer.!ctkframe
        when we try to use e.g. self.frame_widget.winfo_width below. To fix this, we
        set a boolean flag "closed" when the window is closed so that we can skip
        running this function if that is the case.
        """

        if self.closed:
            return

        max_width = self.frame_widget.winfo_width()
        max_height = self.frame_widget.winfo_height()

        # At startup, this area might be of size zero. If so, try later
        if not (max_width > 1 and max_height > 1):
            return

        # Convert the OpenCV image to a CTkImage to display in the widget
        new_image = utils.opencv_to_ctk_image(
            image=self.image, width=max_width, height=max_height
        )
        self.image_widget.configure(image=new_image)

    def _on_window_close(self) -> None:
        """
        Custom window close protocol to set a boolean flag indicating that the window
        has been closed. This is to mitigate a potential race condition/error where
        some underlying widgets are destroyed while we still try to reference them. See
        the function _resize image for more information.
        """
        logger.debug(f"Closed ImageViewer with name '{self.name}' ({self})")
        self.closed = True
        self.destroy()

    def _on_window_resize(self, event: tk.Event) -> None:

        # We need to make sure that the only widget that is allowed to
        # trigger the image resizing is the window itself. Otherwise, when
        # we update the image size, the image widget itself will generate
        # a 'Configure' event which will trigger this function again. That
        # will lead to an endless stream of events.
        if event.widget == self:
            return
        # We are only interested in updating the image size if the window
        # changes size. This event is also triggered when the window moves,
        # so we keep track of the current window size and compare to the new
        # one and update only if it changes.
        new_size = [event.width, event.height]
        if self.current_size == new_size:
            return
        # Modify values inplace to keep the reference intact
        self.current_size[:] = new_size[:]
        # Make sure that the containing frame widget has been updated
        # since it is that size which determines the maximum size of
        # the displayed image inside. In some cases, like when the user
        # expands the window to full-screen, this frame might not have
        # enough time to update before we attempt to update time image
        # within. Then it is not possible to fully expand the image to
        # the entire frame size. By manually calling the update here we
        # can ensure that the frame is the maximum size first.
        self.frame_widget.update()
        self._resize_image()


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
        image: types.Image,
        name: str,
        index: int,
        master: ctk.CTkBaseClass,
        move_entry: t.Callable,
    ):
        super().__init__(master=master)

        self.var_selected = tk.IntVar(value=0)
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.name = name

        self.grid(row=index, column=0, padx=5, pady=5, sticky="nsew")

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
            text=str(index),
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

    def set_current_image(self, image: types.Image) -> None:
        """
        Update the current displayed image of this Entry. This will not modify
        the original OpenCV image stored in this object. This will also update
        the smaller thumbnail picture used in the Entry.
        :param image: The new OpenCV image to set the current displayed image to
        """
        self.current_image = image.copy()
        thumbnail_image = utils.opencv_to_ctk_image(image=image, width=230, height=400)
        self.image_widget.configure(image=thumbnail_image)

    def open_image_viewer_window(self) -> None:
        """
        Open an image viewer window displaying the current image of this Entry.
        """
        ImageViewer(image=self.current_image, name=self.name)
