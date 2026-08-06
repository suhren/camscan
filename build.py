import platform

import PyInstaller.__main__

from camscan import __app_name__, __version__

PLATFORM = platform.system().lower()


def build() -> None:
    """
    Build the application using PyInstaller.

    Some notes:
    
    - To build the application as an executable, you need to also ensure that the
      "pyinstaller" python module has ben installed.
    - Some imports required by the application might not be collected properly by
      pyinstaller. To fix this, provide them as "hidden" imports.
      See https://stackoverflow.com/q/52675162 on the subject
    """

    PyInstaller.__main__.run([
        "camscan/app.py",
        "--onefile",
        "--name",
        f"{__app_name__}-{PLATFORM}-{__version__}",
        "--hidden-import",
        "PIL",
        "--hidden-import",
        "PIL._imagingtk",
        "--hidden-import",
        "PIL._tkinter_finder",
    ])


if __name__ == "__main__":
    build()