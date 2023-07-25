"""
- Camera preview [DONE]
- Button to take image (hot key with space bar) [DONE]
- Checkbox for two-page mode [DONE]
- Camera settings
  - Camera source [DONE]
  - Open camera settings dialogue [DONE] 
- Image settings [DONE]
  - Grayscale [DONE]
  - Sharpen? [DONE]
- List with taken images [DONE]
  - Ability to remove images [DONE]
  - Ability to reorder images? [DONE]
- Export button [DONE]
  - To PDF [DONE]
  - To directory of images [DONE]

TODO:
- Fix bug where the scrollbar doesn't sync up if many entries (around 10) are deleted 
"""

import os
import re
import logging
import functools
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
import customtkinter as ctk
import PIL

import utils
from app.camera import Camera
from app import postprocessing
from camscan import scanner


WINDOW_WIDTH = 1536
WINDOW_HEIGHT = 864

camera = Camera()

LEFT_MENU_PAD_X = 20
LEFT_MENU_PAD_Y = 5

RIGHT_MENU_PAD_X = 10
RIGHT_MENU_PAD_Y = 5

LEFT_MENU_PACK_KWARGS = dict(padx=LEFT_MENU_PAD_X, pady=LEFT_MENU_PAD_Y)
RIGHT_MENU_PACK_KWARGS = dict(padx=RIGHT_MENU_PAD_X, pady=RIGHT_MENU_PAD_Y)

# See the OpenCV documentation for the supported file types to export:
# https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
EXPORT_SEPARATE_FILE_TYPES = [
    "png",
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
]
EXPORT_MERGED_FILE_TYPES = [
    "pdf",
]


POSTPROCESSING_OPTIONS = {
    "None": postprocessing.dummy,
    "Sharpen": postprocessing.sharpen,
    "Grayscale": postprocessing.grayscale,
    "Black and White": postprocessing.black_and_white,
}

RESOLUTIONS = [
    "3264x2448",
    "3264x1836",
    "2592x1944",
    "2048x1536",
    "1920x1080",
    "1600x1200",
    "1280x720",
    "1024x768",
    "800x600",
    "640x480",
]


logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def opencv_to_pil_image(
    image: cv2.Mat,
    width: int = None,
    height: int = None,
) -> PIL.Image:
    return PIL.Image.fromarray(
        utils.resize_with_aspect_ratio(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            width=width,
            height=height,
        )
    )


def opencv_to_tk_image(
    image: cv2.Mat,
    width: int = None,
    height: int = None,
) -> ctk.CTkImage:
    pil_image = opencv_to_pil_image(image=image, width=width, height=height)
    return ctk.CTkImage(
        pil_image,
        size=(pil_image.width, pil_image.height),
    )


class CaptureEntry:
    def __init__(self, image: cv2.Mat, name: str, index: int, master, move_entry):
        self.var_selected = tk.IntVar(value=0)
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.name = name
        self.frame = ctk.CTkFrame(master=master)
        self.frame.grid(row=index, column=0, padx=5, pady=5, sticky="nsew")

        self.move_up_button = ctk.CTkButton(
            master=self.frame,
            text="🔼",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=functools.partial(move_entry, self, -1),
        )
        self.selection_checkbox = ctk.CTkCheckBox(
            master=self.frame,
            text=None,
            checkbox_width=24,
            checkbox_height=24,
            width=24,
            height=24,
            variable=self.var_selected,
        )
        self.move_down_button = ctk.CTkButton(
            master=self.frame,
            text="🔽",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=functools.partial(move_entry, self, 1),
        )
        self.image_widget = ctk.CTkButton(
            master=self.frame,
            fg_color="transparent",
            text=None,
            command=self.open_image_viewer_dialog,
        )
        self.index_label = ctk.CTkLabel(
            master=self.frame,
            text=str(index),
        )
        self.name_label = ctk.CTkLabel(
            master=self.frame,
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
        self.current_image = image.copy()
        thumbnail_image = opencv_to_tk_image(image=image, width=230, height=400)
        self.image_widget.photo = thumbnail_image
        self.image_widget.configure(image=thumbnail_image)

    def open_image_viewer_dialog(self):
        width = self.current_image.shape[1]
        height = self.current_image.shape[0]

        window = ctk.CTkToplevel()
        window.geometry(f"{width}x{height}")
        window.title(self.name)

        ctk.CTkLabel(
            master=window,
            text=None,
            image=opencv_to_tk_image(image=self.current_image),
        ).pack()

        # Make sure this window is on top of the main window
        window.lift()
        window.attributes("-topmost", True)


class CameraScannerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.entries = []
        self.var_postprocessing_option = tk.StringVar(
            value=list(POSTPROCESSING_OPTIONS.keys())[0]
        )
        self.var_two_page_mode = tk.IntVar(value=0)
        self.var_free_capture_mode = tk.IntVar(value=0)
        self.var_select_all_captures = tk.IntVar(value=0)
        self.var_merged_captures_file_type = tk.StringVar(
            value=EXPORT_MERGED_FILE_TYPES[0]
        )
        self.var_separate_captures_file_type = tk.StringVar(
            value=EXPORT_SEPARATE_FILE_TYPES[0]
        )
        self.var_select_all_captures = tk.IntVar(value=0)

        # configure window
        self.title("Scanner")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

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
        self.camera_selection_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Configure Camera",
            command=configure_camera_event,
        )
        self.camera_settings_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Camera Driver Settings",
            command=change_camera_settings_event,
        )

        # Add a menu for the color settings
        self.post_processing_menu_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Postprocessing:", anchor="w"
        )
        self.post_processing_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=list(POSTPROCESSING_OPTIONS.keys()),
            command=self.change_post_processing_event,
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
            values=sorted(EXPORT_SEPARATE_FILE_TYPES),
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
            values=sorted(EXPORT_MERGED_FILE_TYPES),
            variable=self.var_merged_captures_file_type,
            state="readonly",
        )
        self.export_merged_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Export merged file",
            command=self.export_merged_captures,
        )

        # Organize left menu items
        self.left_sidebar_title_label.pack(padx=LEFT_MENU_PAD_X, pady=20)
        self.camera_settings_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.camera_selection_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.camera_settings_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.post_processing_menu_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.post_processing_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.scaling_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.scaling_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.capture_image_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.free_capture_setting_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.two_page_setting_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.capture_image_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_button.pack(**LEFT_MENU_PACK_KWARGS)

        # Configure the central widget showing the camera feed
        self.camera_image_widget = ctk.CTkLabel(self, text=None, padx=0, pady=0)
        self.camera_image_label = ctk.CTkLabel(
            self,
            text="No Camera",
            font=ctk.CTkFont(size=20, weight="bold"),
            padx=0,
            pady=0,
        )

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
            row=0, column=0, columnspan=2, padx=LEFT_MENU_PAD_X, pady=20
        )
        self.select_all_captures_check_box.grid(
            row=1, column=0, **RIGHT_MENU_PACK_KWARGS
        )
        self.delete_captures_button.grid(row=1, column=1, **RIGHT_MENU_PACK_KWARGS)
        self.scrollable_frame.grid(
            row=2, column=0, columnspan=2, sticky="nsew", **RIGHT_MENU_PACK_KWARGS
        )

        # Organize main frames
        self.left_sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.camera_image_widget.grid(row=0, column=1, sticky="nsew")
        self.camera_image_label.grid(row=0, column=1, sticky="nsew")
        self.camera_image_widget.lift()
        self.right_sidebar_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")

        # Hotkeys
        self.bind(sequence="<space>", func=lambda _: self.capture_image())

        self.show_frame()

    def capture(self) -> tuple[cv2.Mat, cv2.Mat, np.ndarray]:
        """
        :return:
            A tuple consisting of the raw image, the extracted warped image, and
            a numpy array describing the contours of the found document. If the
            video capture could not read a frame successfully, return None.
        """
        img_capture = camera.capture()

        if img_capture is not None:
            scan_result = scanner.main(img_capture)
            return (
                img_capture,
                scan_result.warped,
                scan_result.contour,
            )

        return (None, None, None)

    def show_frame(self):
        max_width = self.camera_image_widget.winfo_width()
        max_height = self.camera_image_widget.winfo_height()

        if not (max_width > 1 and max_height > 1):
            # run again after 20ms (0.02s)
            self.after(ms=20, func=self.show_frame)
            return

        raw_image, _, contour = self.capture()

        if raw_image is not None:
            postprocessing_option = self.var_postprocessing_option.get()
            postprocessing_function = POSTPROCESSING_OPTIONS[postprocessing_option]

            image = postprocessing_function(raw_image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if not self.var_free_capture_mode.get():
                image = utils.draw_contour(image=image, contour=contour)

            image_width = image.shape[1]
            image_height = image.shape[0]

            if image_width > max_width or image_height > max_height:
                image = opencv_to_tk_image(
                    image=image, width=max_width, height=max_height
                )
            else:
                image = opencv_to_tk_image(image=image)

            self.camera_image_widget.photo = image
            self.camera_image_widget.configure(image=image)
            self.camera_image_widget.lift()
        else:
            self.camera_image_label.lift()

        # run again after 20ms (0.02s)
        self.after(ms=20, func=self.show_frame)

    def capture_image(self):
        full_image, warped_image, _ = self.capture()

        if self.var_free_capture_mode.get():
            if full_image is not None:
                image = full_image
            else:
                tk.messagebox.showerror(
                    title="Error",
                    message="Could not capture an image from the Camera.",
                )
                return

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

        timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")

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

        self.apply_post_processing(entries=new_entries)
        self.entries += new_entries

        self.scrollable_frame.update()
        self.scrollable_frame._parent_canvas.yview_moveto(1.0)

    def move_entry(self, entry: CaptureEntry, distance: int):
        i = self.entries.index(entry)
        j = i + distance

        if j < 0 or j >= len(self.entries):
            return

        i_grid_row = self.entries[i].frame.grid_info()["row"]
        j_grid_row = self.entries[j].frame.grid_info()["row"]

        logging.debug(f"Switching entries in rows {i_grid_row} and {j_grid_row}")

        # Switch grid positions
        self.entries[i].frame.grid(row=j_grid_row)
        self.entries[j].frame.grid(row=i_grid_row)

        # Switch index labels
        self.entries[i].index_label.configure(text=str(j + 1))
        self.entries[j].index_label.configure(text=str(i + 1))

        # Switch the locations of the entries in the list
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def select_all_entries(self):
        select = self.var_select_all_captures.get()
        for entry in self.entries:
            if select:
                entry.selection_checkbox.select()
            else:
                entry.selection_checkbox.deselect()

    def delete_selected_entries(self):
        entries_to_delete = [e for e in self.entries if e.var_selected.get()]

        logging.debug(f"Removing {len(entries_to_delete)} entries")

        for entry in entries_to_delete:
            entry.frame.destroy()
            self.entries.remove(entry)

        for i, entry in enumerate(self.entries):
            entry.frame.grid(row=i)

        self.scrollable_frame._parent_canvas.yview_moveto(0.0)
        self.scrollable_frame.update()
        self.scrollable_frame._parent_canvas.yview_moveto(1.0)

        self.select_all_captures_check_box.deselect()

    def export_merged_captures(self):
        file_type = self.var_merged_captures_file_type.get()

        n = len(self.entries)

        # If there are no captures, show a message box and return
        if n == 0:
            tk.messagebox.showerror(
                title="Error",
                message="There are no captures to export",
            )
            return

        timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        initialfile = f"captures_{timestamp_str}.{file_type}"

        file_path = tk.filedialog.asksaveasfilename(
            initialfile=initialfile,
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )

        if not file_path:
            return

        # convert to PIL image
        images = [opencv_to_pil_image(entry.current_image) for entry in self.entries]

        first_image = images[0]
        remaining_images = images[1:]

        first_image.save(
            file_path,
            save_all=True,
            append_images=remaining_images,
        )

        tk.messagebox.showinfo(
            title="Export Successful",
            message=f"{n} captures exported as {file_type} to {file_path}",
        )

    def export_separate_captures(self):
        file_type = self.var_separate_captures_file_type.get()

        n = len(self.entries)

        # If there are no captures, show a message box and return
        if n == 0:
            tk.messagebox.showerror(
                title="Error",
                message="There are no captures to export",
            )
            return

        file_dialog_directory = tk.filedialog.askdirectory()

        if not file_dialog_directory:
            return

        timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S")

        output_directory = f"{file_dialog_directory}/captures_{timestamp_str}"

        os.makedirs(output_directory, exist_ok=True)

        for i, entry in enumerate(self.entries, start=1):
            cv2.imwrite(
                filename=f"{output_directory}/{i}_{entry.name}.{file_type}",
                img=entry.current_image,
            )

        tk.messagebox.showinfo(
            title="Export Successful",
            message=f"{n} captures exported as {file_type} to {output_directory}",
        )

    def change_post_processing_event(self, *args):
        self.apply_post_processing(entries=self.entries)

    def apply_post_processing(self, entries: list[CaptureEntry]):
        postprocessing_option = self.var_postprocessing_option.get()
        postprocessing_function = POSTPROCESSING_OPTIONS[postprocessing_option]
        for entry in entries:
            new_image = postprocessing_function(entry.original_image)
            entry.set_current_image(image=new_image)


def configure_camera_event():
    def _set_camera_index(index: int):
        camera.set_index(index=int(index))

    def _update_available_camera_indices():
        available_camera_indices = camera.get_available_device_indices()
        camera_index_combobox.configure(values=list(map(str, available_camera_indices)))
        if available_camera_indices:
            camera_index_combobox.set(value=str(available_camera_indices[0]))
            _set_camera_index(available_camera_indices[0])

    def _set_camera_resolution(resolution_string: str):
        regex = re.compile(r"^(\d+)x(\d+)$")
        matches = regex.findall(resolution_string)
        if not matches:
            tk.messagebox.showerror(
                title="Error",
                message="The resolution string must be on the form '<width>x<height>'",
            )
            return

        resolution = (int(matches[0][0]), int(matches[0][1]))
        camera.set_resolution(resolution=resolution)

    window = ctk.CTkToplevel()
    window.resizable(width=False, height=False)
    window.title("Camera Configuration")

    possible_camera_indices = list(map(str, range(10)))
    var_camera_index = tk.StringVar(value=possible_camera_indices[0])
    current_resolution_string = "x".join(map(str, camera.resolution))
    var_camera_resolution = tk.StringVar(value=current_resolution_string)
    var_custom_camera_resolution = tk.StringVar(value=current_resolution_string)

    camera_index_label = ctk.CTkLabel(
        master=window,
        text="Select Camera Index:",
    )
    camera_index_combobox = ctk.CTkOptionMenu(
        master=window,
        values=possible_camera_indices,
        command=_set_camera_index,
        state="readonly",
        variable=var_camera_index,
    )
    find_camera_indices_button = ctk.CTkButton(
        master=window,
        text="Identify Cameras",
        command=_update_available_camera_indices,
    )
    camera_resolution_label = ctk.CTkLabel(
        master=window,
        text="Camera Resolution:",
    )
    camera_resolution_combobox = ctk.CTkOptionMenu(
        master=window,
        values=RESOLUTIONS,
        command=_set_camera_resolution,
        variable=var_camera_resolution,
    )
    custom_camera_resolution_label = ctk.CTkLabel(
        master=window,
        text="Custom Camera Resolution:",
    )
    custom_camera_resolution_entry = ctk.CTkEntry(
        master=window, textvariable=var_custom_camera_resolution
    )

    def _set_custom_camera_resolution():
        _set_camera_resolution(var_custom_camera_resolution.get())

    custom_camera_resolution_button = ctk.CTkButton(
        master=window,
        text="Set Custom Resolution",
        command=_set_custom_camera_resolution,
    )

    pack_kwargs = dict(padx=10, pady=5)
    camera_index_label.pack(padx=10, pady=(20, 5))
    camera_index_combobox.pack(**pack_kwargs)
    find_camera_indices_button.pack(**pack_kwargs)
    camera_resolution_label.pack(**pack_kwargs)
    camera_resolution_combobox.pack(**pack_kwargs)
    custom_camera_resolution_label.pack(**pack_kwargs)
    custom_camera_resolution_entry.pack(**pack_kwargs)
    custom_camera_resolution_button.pack(padx=10, pady=(5, 20))

    # Make sure this window is on top of the main window
    window.lift()
    window.attributes("-topmost", True)


def change_camera_settings_event():
    camera.show_settings()


def change_ui_appearance_event(new_appearance_mode: str):
    ctk.set_appearance_mode(new_appearance_mode)


def change_ui_scaling_event(new_scaling: str):
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    ctk.set_widget_scaling(new_scaling_float)


if __name__ == "__main__":
    app = CameraScannerApp()
    app.mainloop()
