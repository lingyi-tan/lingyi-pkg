import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lingyi-pkg", # Replace with your own username
    version="0.0.1",
    author="Lingyi Tan",
    author_email="tanlingyi.pku@gmail.com",
    description="Lingyi's utility package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lingyi-tan/lingyi-pkg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
)
