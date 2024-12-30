from setuptools import setup, find_packages

setup(
    name="pymatcal_pytorch", 
    version="0.1.0",         
    author="Fang Han",
    author_email="fanghan@buffalo.edu",
    description="A Python package for PyTorch-based computations and ray tracing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/spebt/pymatcal-pytorch",  
    packages=find_packages(), 
    include_package_data=True, 
    install_requires=[
        "torch",        # Add required dependencies here
        "numpy",
        "h5py",
        "pyyaml"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Specify minimum Python version
)
