import re
import functools

import tkinter as tk
import customtkinter as ctk

from app import config as cfg
from app.camera import Camera
from app.widgets.tooltip import Tooltip


class CameraConfigWindow(ctk.CTkToplevel):
    def __init__(self, master: ctk.CTkToplevel, camera: Camera):
        super().__init__(master=master)

        self.camera = camera

        # Define the variables
        possible_camera_indices = list(map(str, range(10)))
        current_resolution_string = "x".join(map(str, self.camera.resolution))
        self.var_camera_index = tk.StringVar(value=possible_camera_indices[0])
        self.var_camera_resolution = tk.StringVar(value=current_resolution_string)
        self.var_custom_camera_resolution = tk.StringVar(
            value=current_resolution_string
        )

        # Create a new top-level window for the camera configuration
        self.resizable(width=False, height=False)
        self.title("Camera Configuration")

        # Define the widgets
        self.camera_index_label = ctk.CTkLabel(
            master=self,
            text="Select Camera Index:",
        )
        self.camera_index_combobox = ctk.CTkOptionMenu(
            master=self,
            values=possible_camera_indices,
            command=self._set_camera_index,
            state="readonly",
            variable=self.var_camera_index,
        )
        self.find_camera_indices_button = ctk.CTkButton(
            master=self,
            text="Identify Cameras",
            command=self._update_available_camera_indices,
        )
        self.camera_res_label = ctk.CTkLabel(
            master=self,
            text="Camera Resolution:",
        )
        self.camera_res_combobox = ctk.CTkOptionMenu(
            master=self,
            values=cfg.RESOLUTIONS,
            command=self._set_camera_resolution,
            variable=self.var_camera_resolution,
        )
        self.custom_camera_res_label = ctk.CTkLabel(
            master=self,
            text="Custom Camera Resolution:",
        )
        self.custom_camera_res_entry = ctk.CTkEntry(
            master=self, textvariable=self.var_custom_camera_resolution
        )

        self.custom_camera_res_button = ctk.CTkButton(
            master=self,
            text="Set Custom Resolution",
            command=functools.partial(
                self._set_camera_resolution, self.var_custom_camera_resolution.get()
            ),
        )

        # Pack the widgets
        pack_kwargs = dict(padx=10, pady=5)
        self.camera_index_label.pack(padx=10, pady=(20, 5))
        self.find_camera_indices_button.pack(**pack_kwargs)
        self.camera_index_combobox.pack(**pack_kwargs)
        self.camera_res_label.pack(**pack_kwargs)
        self.camera_res_combobox.pack(**pack_kwargs)
        self.custom_camera_res_label.pack(**pack_kwargs)
        self.custom_camera_res_entry.pack(**pack_kwargs)
        self.custom_camera_res_button.pack(padx=10, pady=(5, 20))

        # Add tooltips
        Tooltip(self.camera_index_combobox, cfg.TOOLTIPS["camera_index"])
        Tooltip(self.find_camera_indices_button, cfg.TOOLTIPS["identify_cameras"])
        Tooltip(self.camera_res_combobox, cfg.TOOLTIPS["camera_resolution"])
        Tooltip(self.custom_camera_res_button, cfg.TOOLTIPS["custom_camera_resolution"])

    def _set_camera_index(self, index: str):
        """Callback for changing the camera device index"""
        self.camera.set_index(index=int(index))

    def _update_available_camera_indices(self):
        """Callback for updating the available camera device indices"""
        camera_indices = self.camera.get_available_device_indices()
        self.camera_index_combobox.configure(values=list(map(str, camera_indices)))
        if camera_indices:
            self.camera_index_combobox.set(value=str(camera_indices[0]))
            self._set_camera_index(camera_indices[0])

    def _set_camera_resolution(self, resolution_string: str):
        """Set the camera resolution from a resolution string"""
        regex = re.compile(r"^(\d+)x(\d+)$")
        matches = regex.findall(resolution_string)
        if matches:
            resolution = (int(matches[0][0]), int(matches[0][1]))
            self.camera.set_resolution(resolution=resolution)
        else:
            tk.messagebox.showerror(
                title="Error",
                message="The resolution string must be on the form '<width>x<height>'",
            )

    def show(self):
        """
        Handle the event for configuring the camera. This is done by opening a
        separate window with the available configuration.
        """
        # Make sure this window is on top of the main window
        # We could simply just set topmost to True and leave it at that, but
        # that will prevent the Tooltips from working properly. We can instead
        # set it to topmost temporarily, use grab_set to set focus, and then
        # set topmost back to False. This brings the window to the front.
        # From the documentation it seems that using .lift(aboveThis=self) would
        # work, but I was not able to make that work.
        self.attributes("-topmost", True)
        self.grab_set()
        self.attributes("-topmost", False)
