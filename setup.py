from setuptools import setup, find_packages

def read_readme():
    """Reads the README.md file with UTF-8 encoding."""
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name="plantbrain-fastml",
    version="0.4.5",
    author="Himanshu Bhansali, Himanshu Ranjan",
    author_email="Himanshu.ranjan@algo8.ai, himanshu.bhansali@algo8.ai",
    description="An AutoML package by plantBrain with classification, regression, and forecasting support.",

    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ALGO8AI/plantbrain-fastml.git",
    project_urls={
        "Documentation": "https://github.com/ALGO8AI/plantbrain-fastml.git",
        "Source": "https://github.com/ALGO8AI/plantbrain-fastml.git"
    },
    packages=find_packages(),
    python_requires=">=3.11.0",
    install_requires=[
        "scikit-learn>=1.6.1",
        "pandas>=2.2.3",
        "numpy>=2.1.3",
        "optuna>=4.1.0",
        "matplotlib>=3.10.0"
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    include_package_data=True,
)
