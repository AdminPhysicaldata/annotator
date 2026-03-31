from setuptools import setup, find_packages

setup(
    name="vive_labeler",
    version="0.1.0",
    description="Advanced multimodal data labeling tool for video and HTC VIVE tracking",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "lerobot>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "opencv-python>=4.8.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "pyqtgraph>=0.13.0",
        "PyOpenGL>=3.1.6",
        "PyOpenGL-accelerate>=3.1.6",
        "scipy>=1.11.0",
        "open3d>=0.17.0",
        "PyQt6>=6.5.0",
        "PyQt6-WebEngine>=6.5.0",
        "torch>=2.0.0",
        "huggingface-hub>=0.19.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "vive-labeler=vive_labeler.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
