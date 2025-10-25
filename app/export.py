import os
import datetime

import cv2
import tkinter as tk

from app.utils import opencv_to_pil_image


def export_merged_captures(images: list[cv2.Mat], file_type: str):
    """
    Export all the current captures as a single merged file.
    """
    n = len(images)

    # If there are no captures, show a message box and return
    if n == 0:
        tk.messagebox.showerror(
            title="Error",
            message="There are no captures to export",
        )
        return

    # Create the name of the output file as a timestamp string
    timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    file_name = f"captures_{timestamp_str}.{file_type}"

    # Bring up a dialog asking for the output file path
    file_path = tk.filedialog.asksaveasfilename(
        initialfile=file_name,
        defaultextension=".pdf",
        filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
    )

    # If no output file was chosen (e.g. dialog cancelled), return
    if not file_path:
        return

    # Convert the captured OpenCV images to PIL images
    images = [opencv_to_pil_image(image) for image in images]

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
    tk.messagebox.showinfo(
        title="Export Successful",
        message=f"{n} captures exported as {file_type} to {file_path}",
    )


def export_separate_captures(images: dict[str, cv2.Mat], file_type: str):
    """
    Export all the current captures as separate files in a directory.
    """
    n = len(images)

    # If there are no captures, show a message box and return
    if n == 0:
        tk.messagebox.showerror(
            title="Error",
            message="There are no captures to export",
        )
        return

    # Bring up a dialog asking for the output directory path
    file_dialog_dir = tk.filedialog.askdirectory()

    # If no output directory was chosen (e.g. dialog cancelled), return
    if not file_dialog_dir:
        return

    # Create the name of the output directory as a timestamp string
    timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    output_dir = f"{file_dialog_dir}/captures_{timestamp_str}"
    os.makedirs(output_dir, exist_ok=True)

    # For each capture, write the image to the output directory
    for i, (name, image) in enumerate(images.items(), start=1):
        cv2.imwrite(
            filename=f"{output_dir}/{i}_{name}.{file_type}",
            img=image,
        )

    # Show a message box indicating to the user that the export succeeded
    tk.messagebox.showinfo(
        title="Export Successful",
        message=f"{n} captures exported as {file_type} to {output_dir}",
    )
