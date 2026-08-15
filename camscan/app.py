"""
This is an application used for scanning documents using a camera connected to
your computer, like your webcam. This module specifically implements the GUI
part of the application, as well as the code used to handle, post process, and
export the captured images.
"""

import functools
import os
import tkinter as tk
import typing as t
from tkinter import filedialog as tk_filedialog
from tkinter import messagebox as tk_messagebox

import customtkinter as ctk
import cv2

from camscan import config, utils
from camscan.camera import CameraManager
from camscan.logging import logger
from camscan.model.model import FloatParameter, IntParameter, ModelResult
from camscan.widgets.camera_configuration import CameraConfiguration
from camscan.widgets.captures import CaptureEntry
from camscan.widgets.image_preview import ImagePreview
from camscan.widgets.input import InputFloat, InputInt
from camscan.widgets.tooltip import Tooltip


class CamScanApp(ctk.CTk):
    """
    Application class for CamScan. This defines a CTk Window object containing
    the entire GUI of the application, as well as supporting code.

    Example usage:
        app = CameraScannerApp()
        app.mainloop()
    """

    def __init__(self) -> None:
        super().__init__()

        self.cm: CameraManager = CameraManager()

        self.entries: list[CaptureEntry] = []
        self.var_postprocessing_option = tk.StringVar(
            value=config.DEFAULT_POSTPROCESSING_OPTION
        )
        self.var_model_option = tk.StringVar(value=config.DEFAULT_MODEL_OPTION)
        self.model = config.MODELS[config.DEFAULT_MODEL_OPTION]
        self.var_debug_mode = tk.IntVar(value=0)
        self.var_two_page_mode = tk.IntVar(value=0)
        self.var_free_capture_mode = tk.IntVar(value=0)
        self.var_select_all_captures = tk.IntVar(value=0)
        self.var_merged_captures_file_type = tk.StringVar(
            value=config.EXPORT_MERGED_FILE_TYPES[0]
        )
        self.var_separate_captures_file_type = tk.StringVar(
            value=config.EXPORT_SEPARATE_FILE_TYPES[0]
        )
        self.var_select_all_captures = tk.IntVar(value=0)

        # configure window
        self.title(config.WINDOW_TITLE)
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")

        # Configure the grid layout
        self.grid_columnconfigure((0, 2), weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Configure the left sidebar
        self.left_sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)

        # Add a label to the top of the sidebar
        self.left_sidebar_title_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text="Settings",
            font=ctk.CTkFont(size=20, weight="bold"),
        )

        # Add a button for the camera settings
        self.camera_configuration_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Camera Configuration:", anchor="w"
        )
        self.camera_configuration_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Configure Camera",
            command=self.configure_camera_event,
        )

        # Add a menu for model settings
        self.model_settings_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Model Settings:", anchor="w"
        )
        self.model_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=list(config.MODELS.keys()),
            command=self.change_model_event,
            variable=self.var_model_option,
        )
        self.configure_model_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Configure Model",
            command=self.configure_model_event,
        )

        # Add a menu for the color settings
        self.postprocessing_menu_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Postprocessing:", anchor="w"
        )
        self.postprocessing_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=list(config.POSTPROCESSING_OPTIONS.keys()),
            command=self.change_postprocessing_event,
            variable=self.var_postprocessing_option,
        )

        # Add a menu for the application UI appearance
        self.appearance_mode_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Appearance Mode:", anchor="w"
        )
        self.appearance_mode_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=["System", "Dark", "Light"],
            command=change_ui_appearance_event,
        )
        self.appearance_mode_option_menu.set("System")

        # Add a menu for the application UI scaling
        self.scaling_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="UI Scaling:", anchor="w"
        )
        self.scaling_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=["80%", "90%", "100%", "110%", "120%"],
            command=change_ui_scaling_event,
        )
        self.scaling_option_menu.set("100%")

        # Add a menu for debug mode
        self.debug_mode_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Debug Mode:", anchor="w"
        )
        self.debug_mode_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Debug Mode",
            variable=self.var_debug_mode,
        )

        # Add a button for capturing the screen
        self.capture_image_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Capture Image", anchor="w"
        )
        self.two_page_setting_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Two-page Mode",
            variable=self.var_two_page_mode,
        )
        self.free_capture_setting_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Free Capture Mode",
            variable=self.var_free_capture_mode,
        )
        self.capture_image_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Capture",
            command=self.capture_image,
        )

        # Add a menu for exporting separate captures
        self.export_separate_captures_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Export Separate Files", anchor="w"
        )
        self.export_separate_captures_option_menu = ctk.CTkComboBox(
            master=self.left_sidebar_frame,
            values=sorted(config.EXPORT_SEPARATE_FILE_TYPES),
            variable=self.var_separate_captures_file_type,
            state="readonly",
        )
        self.export_separate_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Export separate files",
            command=self.export_separate_captures,
        )

        # Add a menu for exporting merged captures
        self.export_merged_captures_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Export Merged Files", anchor="w"
        )
        self.export_merged_captures_option_menu = ctk.CTkComboBox(
            master=self.left_sidebar_frame,
            values=sorted(config.EXPORT_MERGED_FILE_TYPES),
            variable=self.var_merged_captures_file_type,
            state="readonly",
        )
        self.export_merged_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Export merged file",
            command=self.export_merged_captures,
        )

        # Organize left menu items
        self.left_sidebar_title_label.pack(padx=config.LEFT_MENU_PAD_X, pady=20)
        self.camera_configuration_label.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.camera_configuration_button.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.model_settings_label.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.model_option_menu.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.configure_model_button.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.postprocessing_menu_label.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.postprocessing_option_menu.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_label.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_option_menu.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.scaling_label.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.scaling_option_menu.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.debug_mode_check_box.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.capture_image_label.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.free_capture_setting_check_box.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.two_page_setting_check_box.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.capture_image_button.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_label.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_option_menu.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_button.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_label.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_option_menu.pack(**config.LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_button.pack(**config.LEFT_MENU_PACK_KWARGS)

        # Configure the central widget showing the camera feed
        self.center_image_preview = ImagePreview(master=self)

        # Configure the right sidebar
        self.right_sidebar_frame = ctk.CTkFrame(self, corner_radius=0)
        self.right_sidebar_frame.grid_rowconfigure((0, 1), weight=0)
        self.right_sidebar_frame.grid_rowconfigure(2, weight=1)

        # Add a label to the top of the sidebar
        self.right_sidebar_title_label = ctk.CTkLabel(
            self.right_sidebar_frame,
            text="Captures",
            font=ctk.CTkFont(size=20, weight="bold"),
        )

        # Create scrollable frame for the captures
        self.scrollable_frame = ctk.CTkScrollableFrame(
            master=self.right_sidebar_frame,
            width=320,
        )
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # Add widgets for selecting all captures and deleting
        self.select_all_captures_check_box = ctk.CTkCheckBox(
            self.right_sidebar_frame,
            text="Select All",
            command=self.select_all_entries,
            variable=self.var_select_all_captures,
        )

        self.delete_captures_button = ctk.CTkButton(
            master=self.right_sidebar_frame,
            text="🗑",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=self.delete_selected_entries,
        )

        # Organize right menu items
        self.right_sidebar_title_label.grid(
            row=0, column=0, columnspan=2, padx=config.LEFT_MENU_PAD_X, pady=20
        )
        self.select_all_captures_check_box.grid(
            row=1, column=0, **config.RIGHT_MENU_PACK_KWARGS
        )
        self.delete_captures_button.grid(
            row=1, column=1, **config.RIGHT_MENU_PACK_KWARGS
        )
        self.scrollable_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="nsew",
            **config.RIGHT_MENU_PACK_KWARGS,
        )

        # Organize main frames
        self.left_sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.center_image_preview.grid(row=0, column=1, sticky="nsew")
        self.right_sidebar_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")

        # Tooltips
        # Left menu
        Tooltip(
            widget=self.camera_configuration_button,
            text=config.TOOLTIPS["camera_configuration"],
        )
        Tooltip(
            widget=self.postprocessing_option_menu,
            text=config.TOOLTIPS["postprocessing"],
        )
        Tooltip(
            widget=self.appearance_mode_option_menu,
            text=config.TOOLTIPS["system_appearance"],
        )
        Tooltip(
            widget=self.scaling_option_menu,
            text=config.TOOLTIPS["system_ui_scaling"],
        )
        Tooltip(
            widget=self.free_capture_setting_check_box,
            text=config.TOOLTIPS["free_capture_mode"],
        )
        Tooltip(
            widget=self.two_page_setting_check_box,
            text=config.TOOLTIPS["two_page_mode"],
        )
        Tooltip(
            widget=self.capture_image_button,
            text=config.TOOLTIPS["capture"],
        )
        Tooltip(
            widget=self.export_separate_captures_button,
            text=config.TOOLTIPS["export_separate"],
        )
        Tooltip(
            widget=self.export_merged_captures_button,
            text=config.TOOLTIPS["export_merged"],
        )
        # Right menu
        Tooltip(
            widget=self.select_all_captures_check_box,
            text=config.TOOLTIPS["select_all"],
        )
        Tooltip(
            widget=self.delete_captures_button,
            text=config.TOOLTIPS["delete"],
        )

        # Hotkeys
        self.bind(sequence=config.CAPTURE_KEYBIND, func=lambda _: self.capture_image())

        self.show_frame()

    def capture(self) -> ModelResult | None:
        """
        Capture an image from the camera and run the document detection
        algorithm on the resulting image.
        :return: A ScanResult or None if we could not read a frame successfully.
        """

        if self.cm.camera is None:
            return None

        img_capture = self.cm.camera.capture()

        if img_capture is not None:
            return self.model.run(img_capture)

        return None

    def show_frame(self) -> None:
        """
        This function is continuously called to show the camera feed in the
        central widget of the application.
        """
        # Get the current width and height of the image preview widget area
        image_preview_width = self.center_image_preview.get_width()
        image_preview_height = self.center_image_preview.get_height()

        # Capture an image and the resulting detected contour from the camera
        result = self.capture()

        info = self.cm.camera.info_string if self.cm.camera is not None else None

        if result is None:
            self.center_image_preview.show(message="No Video", info=info)

        elif result.error_message:
            self.center_image_preview.show(message=result.error_message, info=info)

        elif self.var_debug_mode.get():
            self.center_image_preview.show(
                image=utils.images_in_grid(
                    images=list(result.debug_images.values()),
                    labels=list(result.debug_images.keys()),
                    output_width=image_preview_width,
                    output_height=image_preview_height,
                ),
                message=result.error_message,
                info=info,
            )

        else:
            # Apply the current postprocessing to the image before displaying
            postprocessing_option = self.var_postprocessing_option.get()
            postprocessing_function = config.POSTPROCESSING_OPTIONS[
                postprocessing_option
            ]
            image = postprocessing_function(result.img)

            # If we are using the 'Free Capture' mode, skip drawing the contour
            if not self.var_free_capture_mode.get() and result.contour is not None:
                image = utils.draw_contour(image=image, contour=result.contour)

            self.center_image_preview.show(image=image, info=info)

        # Run again after a delay
        self.after(ms=config.CAMERA_WAIT_MS, func=self.show_frame)

    def capture_image(self) -> None:
        """
        Capture an image using the camera.
        """
        result = self.capture()

        if result is None or result.img is None:
            tk_messagebox.showerror(
                title="Error",
                message="Could not capture an image from the Camera.",
            )
            return

        # If we are using Free Capture mode, use the full uncropped image
        if self.var_free_capture_mode.get():
            image = result.img

        # Otherwise, use the warped cropped extracted image
        elif result.warped is not None:
            image = result.warped
        else:
            tk_messagebox.showerror(
                title="Error",
                message=(
                    "Could not extract the document image from the Camera. "
                    "Enable 'Free Capture Mode' to take the image anyway."
                ),
            )
            return

        # Give the capture a name using a timestamp string
        timestamp_str = utils.get_timestamp_str()

        # If we are using two-page mode, cut the image into left and right parts
        if self.var_two_page_mode.get():
            cutoff_width = image.shape[1] // 2
            left_image = image[:, :cutoff_width]
            right_image = image[:, cutoff_width:]
            new_entries = [
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=left_image,
                    name=f"{timestamp_str}_1",
                    index=len(self.entries) + 1,
                    move_entry=self.move_entry,
                ),
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=right_image,
                    name=f"{timestamp_str}_2",
                    index=len(self.entries) + 2,
                    move_entry=self.move_entry,
                ),
            ]
        # Otherwise, take the entire image and as as an entry
        else:
            new_entries = [
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=image,
                    name=timestamp_str,
                    index=len(self.entries) + 1,
                    move_entry=self.move_entry,
                )
            ]

        # If a postprocessing function is selected, apply it to the new images
        self.apply_postprocessing(entries=new_entries)
        self.entries += new_entries

        # Update the scrollable frame with the entries and move it to the bottom
        self.scrollable_frame.update()
        self.scrollable_frame._parent_canvas.yview_moveto(1.0)

    def move_entry(self, entry: CaptureEntry, distance: int) -> None:
        """
        Move an entry in the capture list either up or down by some distance.
        :param entry: The CaptureEntry to move
        :param distance: The move distance (-1 to move up, or +1 to move down)
        """
        # Find the current index 'i' of the entry and the destination index 'j'
        i = self.entries.index(entry)
        j = i + distance

        # If the destination index is out of range, skip the operation
        if j < 0 or j >= len(self.entries):
            return

        # Get the current grid rows of the entries. This is not really needed
        # since the indices i and j should be the same as the grid row
        i_grid_row = self.entries[i].frame.grid_info()["row"]
        j_grid_row = self.entries[j].frame.grid_info()["row"]

        # Switch grid positions
        logger.debug(f"Switching entries in rows {i_grid_row} and {j_grid_row}")
        self.entries[i].frame.grid(row=j_grid_row)
        self.entries[j].frame.grid(row=i_grid_row)

        # Switch index labels
        self.entries[i].index_label.configure(text=str(j + 1))
        self.entries[j].index_label.configure(text=str(i + 1))

        # Switch the locations of the entries in the list
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def select_all_entries(self) -> None:
        """
        Select or deselect all current capture entries.
        """
        # Depending on the state of the checkbox, select or deselect all entries
        select = self.var_select_all_captures.get()
        for entry in self.entries:
            if select:
                entry.selection_checkbox.select()
            else:
                entry.selection_checkbox.deselect()

    def delete_selected_entries(self) -> None:
        """
        Delete all the currently selected capture entries.
        """
        # Select the entries based on the state of their checkbox variable
        entries_to_delete = [e for e in self.entries if e.var_selected.get()]
        logger.debug(f"Removing {len(entries_to_delete)} entries")

        # For each such entry, destroy its frame and remove from the list
        for entry in entries_to_delete:
            entry.frame.destroy()
            self.entries.remove(entry)

        # After deletion, update the grid positions of the remaining entries
        for i, entry in enumerate(self.entries):
            entry.frame.grid(row=i)

        # There is some peculiar behavior of the scrollbar in the scrollable
        # frame when all entries are deleted at once. If there are enough
        # entries (around 5+) to make the scrollbar active, and it is scrolled
        # all the way to the bottom, it will not correctly update its allowed
        # range of scrolling when the entries are deleted. Instead, it will
        # still be scrolled all the way to the bottom, with the scrollable frame
        # being completely empty. After testing, it seems that one (hacky)
        # solution to this is to do the following:
        # - Add back a widget in the grid (a dummy frame in this solution)
        # - Move the scroll all the way back up to the top (yview_moveto)
        # - Call the update function on the scrollable frame
        # - Delete the dummy frame after it is no longer needed.
        # By adding this dummy widget, it seems to make the update of the
        # scrollable frame also update the scrollbar to the correct range.
        # Without it, this does not work!
        if len(self.entries) == 0:
            dummy_frame = ctk.CTkFrame(master=self.scrollable_frame)
            dummy_frame.grid(row=0, column=0)
            self.scrollable_frame._parent_canvas.yview_moveto(0.0)
            self.scrollable_frame.update()
            dummy_frame.destroy()

        # Uncheck the checkbox for selecting all entries
        self.select_all_captures_check_box.deselect()

    def export_merged_captures(self) -> None:
        """
        Export all the current captures as a single merged file.
        """
        # Get the currently select file type to export as
        file_type = self.var_merged_captures_file_type.get()

        n = len(self.entries)

        # If there are no captures, show a message box and return
        if n == 0:
            tk_messagebox.showerror(
                title="Error",
                message="There are no captures to export",
            )
            return

        # Create the name of the output file as a timestamp string
        timestamp_str = utils.get_timestamp_str()
        initialfile = f"captures_{timestamp_str}.{file_type}"

        # Bring up a dialog asking for the output file path
        file_path = tk_filedialog.asksaveasfilename(
            initialfile=initialfile,
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )

        # If no output file was chosen (e.g. dialog cancelled), return
        if not file_path:
            return

        # Convert the captured OpenCV images to PIL images
        images = [utils.opencv_to_pil_image(e.current_image) for e in self.entries]

        # The PIL save functionality requires that we initiate it from a single
        # image, then append the remaining images as function parameter
        first_image = images[0]
        remaining_images = images[1:]
        first_image.save(
            file_path,
            save_all=True,
            append_images=remaining_images,
        )

        # Show a message box indicating to the user that the export succeeded
        tk_messagebox.showinfo(
            title="Export Successful",
            message=f"{n} captures exported as {file_type} to {file_path}",
        )

    def export_separate_captures(self) -> None:
        """
        Export all the current captures as separate files in a directory.
        """
        # Get the currently select file type to export as
        file_type = self.var_separate_captures_file_type.get()

        n = len(self.entries)

        # If there are no captures, show a message box and return
        if n == 0:
            tk_messagebox.showerror(
                title="Error",
                message="There are no captures to export",
            )
            return

        # Bring up a dialog asking for the output directory path
        file_dialog_dir = tk_filedialog.askdirectory()

        # If no output directory was chosen (e.g. dialog cancelled), return
        if not file_dialog_dir:
            return

        # Create the name of the output directory as a timestamp string
        timestamp_str = utils.get_timestamp_str()
        output_dir = f"{file_dialog_dir}/captures_{timestamp_str}"
        os.makedirs(output_dir, exist_ok=True)

        # For each capture, write the image to the output directory
        for i, entry in enumerate(self.entries, start=1):
            cv2.imwrite(
                filename=f"{output_dir}/{i}_{entry.name}.{file_type}",
                img=entry.current_image,
            )

        # Show a message box indicating to the user that the export succeeded
        tk_messagebox.showinfo(
            title="Export Successful",
            message=f"{n} captures exported as {file_type} to {output_dir}",
        )

    def change_postprocessing_event(self, *args: t.Any) -> None:
        """
        Handle the event when the chose postprocessing function changes.
        When it does, apply it to all current capture entries.
        """
        self.apply_postprocessing(entries=self.entries)

    def apply_postprocessing(self, entries: list[CaptureEntry]) -> None:
        """
        Apply currently chosen postprocessing function to given capture entries.
        :param entries: The capture entries to apply the postprocessing to
        """
        postprocessing_option = self.var_postprocessing_option.get()
        postprocessing_function = config.POSTPROCESSING_OPTIONS[postprocessing_option]
        for entry in entries:
            new_image = postprocessing_function(entry.original_image)
            entry.set_current_image(image=new_image)

    def change_model_event(self, *args: t.Any) -> None:
        self.model = config.MODELS[self.var_model_option.get()]

    def configure_camera_event(self) -> None:
        """
        Handle the event for configuring the camera. This is done by opening a
        separate window with the available configuration.
        """
        if self.cm.camera is not None:
            CameraConfiguration(master=self, cm=self.cm)

    def configure_model_event(self) -> None:
        """
        Handle the event for configuring the model. This is done by opening a
        separate window with the available configuration.
        """

        # Create a new top-level window for the model configuration
        window = ctk.CTkToplevel()
        window.resizable(width=False, height=False)
        window.title("Model Configuration")

        def _on_value(val: t.Any, name: str) -> None:
            self.model.param(name).set(val)

        for p in self.model.parameters:
            if isinstance(p, IntParameter):
                InputInt(
                    master=window,
                    label=p.name,
                    value=p.value,
                    min_value=p.min_value,
                    max_value=p.max_value,
                    default_value=p.default_value,
                    on_value=functools.partial(_on_value, name=p.name),
                )

            elif isinstance(p, FloatParameter):
                InputFloat(
                    master=window,
                    label=p.name,
                    value=p.value,
                    min_value=p.min_value,
                    max_value=p.max_value,
                    default_value=p.default_value,
                    on_value=functools.partial(_on_value, name=p.name),
                )

        # Make sure this window is on top of the main window
        # We could simply just set topmost to True and leave it at that, but
        # that will prevent the Tooltips from working properly. We can instead
        # set it to topmost temporarily, use grab_set to set focus, and then
        # set topmost back to False. This brings the window to the front.
        # From the documentation it seems that using .lift(aboveThis=self) would
        # work, but I was not able to make that work.
        window.attributes("-topmost", True)
        window.grab_set()
        window.attributes("-topmost", False)


def change_ui_appearance_event(new_appearance_mode: str) -> None:
    """
    Handle the event to update the application appearance.
    :param new_appearance_mode: The appearance mode (System, Dark, Light)
    """
    ctk.set_appearance_mode(new_appearance_mode)


def change_ui_scaling_event(new_scaling: str) -> None:
    """
    Handle the event to update the application UI scale.
    :param new_scaling: The new scaling string on the form XX%
    """
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    ctk.set_widget_scaling(new_scaling_float)


if __name__ == "__main__":
    app = CamScanApp()
    app.mainloop()
