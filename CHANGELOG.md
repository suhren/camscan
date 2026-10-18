# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.0] - 2026-10-18

### Added

- **Image as Input feature**: Users can now process existing photos instead of only using live camera feed
  - **File Mode**: Load and process single image files
  - **Folder Mode**: Batch process all images in a folder (recursive subfolder support)
  - **Input Source Selector**: Dropdown menu to switch between Camera/File/Folder modes
- **Per-step timing visualization**: Debug mode now shows execution time for each processing step with percentages
- **New document detection models based on GrabCut algorithm**:
  - `GrabCutContours`: Uses GrabCut with contour finding for reliable document detection
  - `GrabCutHough`: Uses GrabCut with Hough line transform for structured documents
- `SUPPORTED_IMAGE_EXTENSIONS` constant in config for centralized image type management
- Dedicated development resources folder with Jupyter notebook for GrabCut experimentation

### Changed

- Updated Python version requirement to >= 3.12 (previously 3.11)

## [v0.2.1] - 2025-10-27

### Added

- Platform-specific requirements files (`requirements_linux.txt`, `requirements_windows.txt`)
- `pyproject.toml` for project configuration
- `camscan/` Python package structure

### Changed

- Reorganized project: moved modules from `app/` to `camscan/` package
- Added app name and version in `__init__.py`
- Updated README with Linux environment fixes
- GitHub workflows now use platform-specific requirements
- Applied black code formatting

### Removed

- Generic `requirements.txt` (replaced by platform-specific files)

## [v0.2.0] - 2023-07-30

Changes:

- Fixed edge-case bug where a numpy array could not be squeezed (https://github.com/suhren/camscan/issues/1)
- Removed unnecessary error logging when no camera image could be taken (https://github.com/suhren/camscan/issues/2)
- Added 'Free Capture Mode' which enables the user to take capture an image even if no document is detected
- Added tooltips when the mouse cursor hovers over widgets in the application interface
- Fixed an issue where the scrollbar would not reset properly when many capture entries were deleted at once
- Updated the image viewer window when a capture is pressed to scale the image properly and fill out the window
- Added tests for the contour finding algorithm to evaluate it on a few test images

## [v0.1.0] - 2023-07-25

This is an initial test release of the software.

[v0.1.0]: https://github.com/suhren/camscan/releases/tag/v0.1.0
[v0.2.0]: https://github.com/suhren/camscan/releases/tag/v0.2.0
[v0.2.1]: https://github.com/suhren/camscan/releases/tag/v0.2.1
[v0.3.0]: https://github.com/suhren/camscan/releases/tag/v0.3.0
