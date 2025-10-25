"""
This is an application used for scanning documents using a camera connected to
your computer, like your webcam. This module specifically implements the GUI
part of the application, as well as the code used to handle, post process, and
export the captured images.
"""

from datetime import datetime
import logging

import customtkinter as ctk
import cv2
import numpy as np
import tkinter as tk

from app import config as cfg
from app.camera import Camera
from app.widgets.tooltip import Tooltip
from app.widgets.capture_entry import CaptureEntry
from app.widgets.camera_feed import CameraFeed
from app.widgets.camera_config import CameraConfigWindow
from app.export import export_merged_captures, export_separate_captures

from camscan import scanner
from camscan.utils import draw_contour


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.ERROR,
)


class CamScanApp(ctk.CTk):
    """
    Application class for CamScan. This defines a CTk Window object containing
    the entire GUI of the application, as well as supporting code.

    Example usage:
        app = CameraScannerApp()
        app.mainloop()
    """

    def __init__(self):
        super().__init__()

        self.camera = Camera()
        self.entries = []
        self.var_postprocessing_option = tk.StringVar(
            value=list(cfg.POSTPROCESSING_OPTIONS.keys())[0]
        )
        self.var_two_page_mode = tk.IntVar(value=0)
        self.var_free_capture_mode = tk.IntVar(value=0)
        self.var_select_all_captures = tk.IntVar(value=0)
        self.var_merged_captures_file_type = tk.StringVar(
            value=cfg.EXPORT_MERGED_FILE_TYPES[0]
        )
        self.var_separate_captures_file_type = tk.StringVar(
            value=cfg.EXPORT_SEPARATE_FILE_TYPES[0]
        )
        self.var_select_all_captures = tk.IntVar(value=0)

        # configure window
        self.title(cfg.APP_NAME)
        self.geometry(f"{cfg.WINDOW_WIDTH}x{cfg.WINDOW_HEIGHT}")

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
        self.camera_settings_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Camera Settings:", anchor="w"
        )
        self.configure_camera_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Configure Camera",
            command=self.configure_camera_event,
        )
        self.camera_settings_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Camera Driver Settings",
            command=self.camera.show_settings,
        )

        # Add a menu for the color settings
        self.postprocessing_menu_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Postprocessing:", anchor="w"
        )
        self.postprocessing_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=list(cfg.POSTPROCESSING_OPTIONS.keys()),
            command=self.change_postprocessing_event,
            variable=self.var_postprocessing_option,
        )

        # Add a menu for the application UI appearance
        self.appearance_mode_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Appearance Mode:", anchor="w"
        )
        self.appearance_mode_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=cfg.UI_APPEARANCE_MODES,
            command=change_ui_appearance_event,
        )
        self.appearance_mode_option_menu.set("System")

        # Add a menu for the application UI scaling
        self.scaling_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="UI Scaling:", anchor="w"
        )
        self.scaling_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=cfg.UI_SCALING_OPTIONS,
            command=change_ui_scaling_event,
        )
        self.scaling_option_menu.set("100%")

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
            values=sorted(cfg.EXPORT_SEPARATE_FILE_TYPES),
            variable=self.var_separate_captures_file_type,
            state="readonly",
        )
        self.export_separate_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Export separate files",
            command=self.export_separate_captures_event,
        )

        # Add a menu for exporting merged captures
        self.export_merged_captures_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Export Merged Files", anchor="w"
        )
        self.export_merged_captures_option_menu = ctk.CTkComboBox(
            master=self.left_sidebar_frame,
            values=sorted(cfg.EXPORT_MERGED_FILE_TYPES),
            variable=self.var_merged_captures_file_type,
            state="readonly",
        )
        self.export_merged_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Export merged file",
            command=self.export_merged_captures_event,
        )

        # Organize left menu items
        self.left_sidebar_title_label.pack(padx=cfg.LEFT_MENU_PAD_X, pady=20)
        self.camera_settings_label.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.configure_camera_button.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.camera_settings_button.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.postprocessing_menu_label.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.postprocessing_option_menu.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_label.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_option_menu.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.scaling_label.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.scaling_option_menu.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.capture_image_label.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.free_capture_setting_check_box.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.two_page_setting_check_box.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.capture_image_button.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_label.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_option_menu.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_button.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_label.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_option_menu.pack(**cfg.LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_button.pack(**cfg.LEFT_MENU_PACK_KWARGS)

        # Configure the central widget showing the camera feed
        self.camera_feed_widget = CameraFeed(master=self)

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

        # Add widgets for selcting all captures and deleting
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
            row=0, column=0, columnspan=2, padx=cfg.LEFT_MENU_PAD_X, pady=20
        )
        self.select_all_captures_check_box.grid(
            row=1, column=0, **cfg.RIGHT_MENU_PACK_KWARGS
        )
        self.delete_captures_button.grid(row=1, column=1, **cfg.RIGHT_MENU_PACK_KWARGS)
        self.scrollable_frame.grid(
            row=2, column=0, columnspan=2, sticky="nsew", **cfg.RIGHT_MENU_PACK_KWARGS
        )

        # Organize main frames
        self.left_sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.camera_feed_widget.grid(row=0, column=1, sticky="nsew")
        self.right_sidebar_frame.grid(row=0, column=2, sticky="nsew")

        # Tooltips
        # Left menu
        Tooltip(self.configure_camera_button, cfg.TOOLTIPS["camera_configuration"])
        Tooltip(self.camera_settings_button, cfg.TOOLTIPS["camera_driver_settings"])
        Tooltip(self.postprocessing_option_menu, cfg.TOOLTIPS["postprocessing"])
        Tooltip(self.appearance_mode_option_menu, cfg.TOOLTIPS["system_appearance"])
        Tooltip(self.scaling_option_menu, cfg.TOOLTIPS["system_ui_scaling"])
        Tooltip(self.free_capture_setting_check_box, cfg.TOOLTIPS["free_capture_mode"])
        Tooltip(self.two_page_setting_check_box, cfg.TOOLTIPS["two_page_mode"])
        Tooltip(self.capture_image_button, cfg.TOOLTIPS["capture"])
        Tooltip(self.export_separate_captures_button, cfg.TOOLTIPS["export_separate"])
        Tooltip(self.export_merged_captures_button, cfg.TOOLTIPS["export_merged"])
        # Right menu
        Tooltip(self.select_all_captures_check_box, cfg.TOOLTIPS["select_all"])
        Tooltip(self.delete_captures_button, cfg.TOOLTIPS["delete"])

        # Hotkeys
        self.bind(sequence=cfg.CAPTURE_KEYBIND, func=lambda _: self.capture_image())

        self.show_frame()

    def capture(self) -> tuple[cv2.Mat, cv2.Mat, np.ndarray]:
        """
        Capture an image from the camera and run the document detection
        algorithm on the resulting image.
        :return:
            A tuple consisting of the raw image, the extracted warped image, and
            a numpy array describing the contours of the found document. If the
            video capture could not read a frame successfully, return None.
        """
        img_capture = self.camera.capture()

        if img_capture is not None:
            scan_result = scanner.main(img_capture)
            return (
                img_capture,
                scan_result.warped,
                scan_result.contour,
            )

        return (None, None, None)

    def show_frame(self):
        """
        This function is continuously called to show the camera feed in the
        central widget of the application.
        """
        # Capture an image and the resulting detected contour from the camera
        image, _, contour = self.capture()

        if image is not None:
            # Apply the current postprocessing to the image before displaying
            postprocessing_option = self.var_postprocessing_option.get()
            postprocessing_function = cfg.POSTPROCESSING_OPTIONS[postprocessing_option]
            image = postprocessing_function(image)
            # The image must have three color channels, so convert if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # If we are using the 'Free Capture' mode, skip drawing the contour
            if not self.var_free_capture_mode.get():
                image = draw_contour(image=image, contour=contour)

        self.camera_feed_widget.display(image=image)

        # Run again after a delay
        self.after(ms=cfg.CAMERA_FEED_WAIT_MS, func=self.show_frame)

    def capture_image(self):
        """
        Capture an image using the camera.
        """
        full_image, warped_image, _ = self.capture()

        # If we are using Free Capture mode, use the full uncropped image
        if self.var_free_capture_mode.get():
            if full_image is not None:
                image = full_image
            else:
                tk.messagebox.showerror(
                    title="Error",
                    message="Could not capture an image from the Camera.",
                )
                return
        # Otherwise, use the warped cropped extracted image
        elif warped_image is not None:
            image = warped_image
        else:
            tk.messagebox.showerror(
                title="Error",
                message=(
                    "Could not extract the document image from the Camera. "
                    "Enable 'Free Capture Mode' to take the image anyway."
                ),
            )
            return

        # Give the capture a name using a timestamp string
        timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")

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
        for entry in new_entries:
            self.entries.append(entry)
            entry.grid(row=entry.index, column=0, padx=5, pady=5, sticky="nsew")

        # Update the scrollable frame with the entries and move it to the bottom
        self.scrollable_frame.update()
        self.scrollable_frame._parent_canvas.yview_moveto(1.0)

    def move_entry(self, entry: CaptureEntry, distance: int):
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
        logging.debug(f"Switching entries in rows {i_grid_row} and {j_grid_row}")
        self.entries[i].frame.grid(row=j_grid_row)
        self.entries[j].frame.grid(row=i_grid_row)

        # Switch index labels
        self.entries[i].index_label.configure(text=str(j + 1))
        self.entries[j].index_label.configure(text=str(i + 1))

        # Switch the locations of the entries in the list
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def select_all_entries(self):
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

    def delete_selected_entries(self):
        """
        Delete all the currently selected capture entries.
        """
        # Select the entries based on the state of their checkbox variable
        entries_to_delete = [e for e in self.entries if e.var_selected.get()]
        logging.debug(f"Removing {len(entries_to_delete)} entries")

        # For each such entry, destroy its frame and remove from the list
        for entry in entries_to_delete:
            entry.destroy()
            self.entries.remove(entry)

        # After deletion, update the grid positions of the remaining entries
        for i, entry in enumerate(self.entries):
            entry.grid(row=i)

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

    def export_merged_captures_event(self):
        """
        Export all the current captures as a single merged file.
        """
        export_merged_captures(
            images=[entry.current_image for entry in self.entries],
            file_type=self.var_merged_captures_file_type.get(),
        )

    def export_separate_captures_event(self):
        """
        Export all the current captures as separate files in a directory.
        """
        export_separate_captures(
            images={entry.name: entry.current_image for entry in self.entries},
            file_type=self.var_separate_captures_file_type.get(),
        )

    def change_postprocessing_event(self, *args):
        """
        Handle the event when the chose postprocessing function changes.
        When it does, apply it to all current capture entries.
        """
        self.apply_postprocessing(entries=self.entries)

    def apply_postprocessing(self, entries: list[CaptureEntry]):
        """
        Apply currently chosen postprocessing function to given capture entries.
        :param entries: The capture entries to apply the postprocessing to
        """
        postprocessing_option = self.var_postprocessing_option.get()
        postprocessing_function = cfg.POSTPROCESSING_OPTIONS[postprocessing_option]
        for entry in entries:
            new_image = postprocessing_function(entry.original_image)
            entry.set_current_image(image=new_image)

    def configure_camera_event(self):
        """
        Handle the event for configuring the camera. This is done by opening a
        separate window with the available configuration.
        """
        CameraConfigWindow(master=self, camera=self.camera).show()


def change_ui_appearance_event(new_appearance_mode: str):
    """
    Handle the event to update the application appearance.
    :param new_appearance_mode: The appearance mode (System, Dark, Light)
    """
    ctk.set_appearance_mode(new_appearance_mode)


def change_ui_scaling_event(new_scaling: str):
    """
    Handle the event to update the application UI scale.
    :param new_scaling: The new scaling string on the form XX%
    """
    ctk.set_widget_scaling(int(new_scaling.replace("%", "")) / 100)


if __name__ == "__main__":
    app = CamScanApp()
    app.mainloop()
