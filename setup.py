import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="food-aesthetics",
    version="0.0.1",
    description="Inferring Aesthetics from Food Pictures",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Alessandro Gambetti",
    author_email="gambetti.alessandro@novasbe.pt",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=["food-aesthetics"],
    include_package_data=True,
    install_requires=["tensorflow", "numpy", "Pillow"],
)
