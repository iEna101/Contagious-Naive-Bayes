import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Contagious Naive Bayes", # Replace with your own username
    version="1.0.0",
    author="Iena Petronella Derks",
    author_email="<inekederks1@gmail.com>",
    description="The package enables the use of Contagious Naive Bayes to perform classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iEna101/Contagious-NB",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
