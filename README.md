# Photosphere Vortex

Photosphere Vortex is a Python-based image processing project focused on creating panoramic images from overlapping frames and improving image color balance for real-world camera captures.

This project combines two main ideas:

- Panorama generation using feature matching and image stitching
- Basic image enhancement using white balance correction

The goal is to create a simple toolset for building wide-angle or stitched photos from multiple images and improving image quality before further editing or use.

## Project Overview

The project currently includes two scripts:

- `photosphere.py` — stitches multiple overlapping images into one result using OpenCV feature detection and homography-based warping
- `enhancement.py` — corrects white balance using a gray-world assumption to improve color consistency in an image

## Features

- SIFT-based image feature detection
- Keypoint matching between frames
- Homography estimation for overlapping image alignment
- Perspective transformation and warping
- Simple white-balance enhancement for camera frames
- Output image preview via OpenCV

## Folder Structure

```text
Photosphere_vortex/
├── photosphere.py
├── enhancement.py
├── frames/
├── img1.png
├── img2.png
├── img3.png
├── img4.png
├── README.md
└── sample captured frames / output images
```

## Requirements

Install the required Python packages:

```bash
pip install opencv-python numpy matplotlib
```

## Running the Panorama Stitching Script

1. Place your images in the project folder or update the filenames in `photosphere.py`.
2. Open a terminal in the project directory.
3. Run:

```bash
python photosphere.py
```

This script reads images like `img1.png`, `img2.png`, and so on, aligns them using matching features, and displays the stitched result.

## Running the Enhancement Script

1. Make sure the target image is present in the project directory.
2. Update the file name in `enhancement.py` if needed.
3. Run:

```bash
python enhancement.py
```

This script applies white-balance correction to the image and saves the enhanced version back to the same file.

## Notes

- This project is intended as a practical experimental image-processing project.
- The panorama pipeline is a basic implementation and may need tuning depending on image overlap, lighting, and camera movement.
- The enhancement step is a lightweight correction method and works best for neutral lighting conditions.

## Possible Improvements

- Add automatic frame selection for panorama input
- Improve stitching quality with better blending and seam removal
- Add image quality controls and output file naming
- Support video frames from RTSP or camera streams
- Create a GUI or web interface for easier use

## License

This project does not currently include a formal license file. If you plan to share it publicly, consider adding an appropriate open-source license such as MIT.

## Author

This project is a personal image processing and computer vision experiment for panoramic image creation and enhancement.
