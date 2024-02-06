from setuptools import setup, find_packages

setup(
    name="satlaspretrain_models",
    version="0.1",
    author="Satlas @ AI2",
    author_email="satlas@allenai.org",
    description="A simple package that makes it easy to load remote sensing foundation models for downstream use cases.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/allenai/satlaspretrain_models",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.1.0',
        'torchvision>=0.16.0',
        'requests'
    ],
)

