import tkinter as tk
import typing as t
from tkinter import messagebox as tk_messagebox

import customtkinter as ctk

from camscan import config
from camscan.camera import CameraManager
from camscan.widgets.camera_settings import CameraSettings
from camscan.widgets.tooltip import Tooltip


class CameraConfiguration(ctk.CTkToplevel):
    """
    Configuration for a camera.
    See https://docs.opencv.org/3.4.20/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    """

    def __init__(
        self,
        master: t.Any,
        cm: CameraManager,
    ) -> None:
        super().__init__(master=master)
        self.resizable(width=False, height=False)
        self.title("Camera Configuration")

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

        # Define the variables
        current_resolution_string = (
            cm.camera.get_resolution_string() if cm.camera is not None else None
        )
        var_camera_name = tk.StringVar(value=cm.camera.name if cm.camera else None)
        var_camera_resolution = tk.StringVar(value=current_resolution_string)
        var_custom_camera_resolution = tk.StringVar(value=current_resolution_string)

        def _set_camera_event(name: str) -> None:
            cm.set_camera(name)

        def _identify_cameras_event() -> None:
            """Callback for updating the available cameras"""
            cm.update_available_cameras()
            camera_name_combobox.configure(values=cm.get_camera_names())
            if cm.camera is None:
                cm.set_camera_to_first_usable()

        def _set_camera_resolution_event(value: str | tuple[int, int]) -> None:
            if cm.camera is None:
                tk_messagebox.showerror(title="Error", message="No camera available")
                return

            try:
                cm.camera.set_resolution(value)
            except ValueError as e:
                tk_messagebox.showerror(title="Error", message=str(e))

        def _show_camera_settings_event() -> None:
            if cm.camera is not None:
                CameraSettings(master=self, camera=cm.camera)

        # Define the widgets
        camera_name_label = ctk.CTkLabel(
            master=self,
            text="Select Camera:",
        )
        camera_name_combobox = ctk.CTkOptionMenu(
            master=self,
            values=cm.get_camera_names(),
            command=_set_camera_event,
            state="readonly",
            variable=var_camera_name,
        )
        identify_cameras_button = ctk.CTkButton(
            master=self,
            text="Identify Available Cameras",
            command=_identify_cameras_event,
        )
        camera_settings_button = ctk.CTkButton(
            master=self,
            text="Camera Settings",
            command=_show_camera_settings_event,
        )
        camera_resolution_label = ctk.CTkLabel(
            master=self,
            text="Camera Resolution:",
        )
        camera_resolution_combobox = ctk.CTkOptionMenu(
            master=self,
            values=config.RESOLUTIONS,
            command=_set_camera_resolution_event,
            variable=var_camera_resolution,
        )
        custom_camera_resolution_label = ctk.CTkLabel(
            master=self,
            text="Custom Camera Resolution:",
        )
        custom_camera_resolution_entry = ctk.CTkEntry(
            master=self, textvariable=var_custom_camera_resolution
        )

        custom_camera_resolution_button = ctk.CTkButton(
            master=self,
            text="Set Custom Resolution",
            command=lambda: _set_camera_resolution_event(
                var_custom_camera_resolution.get()
            ),
        )

        # Pack the widgets
        pack_kwargs = {"padx": 10, "pady": 5}
        camera_name_label.pack(padx=10, pady=(20, 5))
        identify_cameras_button.pack(**pack_kwargs)
        camera_name_combobox.pack(**pack_kwargs)
        camera_settings_button.pack(**pack_kwargs)
        camera_resolution_label.pack(**pack_kwargs)
        camera_resolution_combobox.pack(**pack_kwargs)
        custom_camera_resolution_label.pack(**pack_kwargs)
        custom_camera_resolution_entry.pack(**pack_kwargs)
        custom_camera_resolution_button.pack(padx=10, pady=(5, 20))

        # Add tooltips
        Tooltip(
            widget=camera_name_combobox,
            text=config.TOOLTIPS["camera_name"],
        )
        Tooltip(
            widget=identify_cameras_button,
            text=config.TOOLTIPS["identify_cameras"],
        )
        Tooltip(
            widget=camera_settings_button,
            text=config.TOOLTIPS["camera_driver_settings"],
        )
        Tooltip(
            widget=camera_resolution_combobox,
            text=config.TOOLTIPS["camera_resolution"],
        )
        Tooltip(
            widget=custom_camera_resolution_button,
            text=config.TOOLTIPS["custom_camera_resolution"],
        )
