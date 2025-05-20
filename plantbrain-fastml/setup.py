from setuptools import setup, find_packages

setup(
    name="plantbrain-fastml",
    version="0.1.0",
    author="Himanshu Bhansali, Himanshu Ranjan",
    author_email="Himanshu.ranjan@algo8.ai, himanshu.bhansali@algo8.ai",
    description="An AutoML package by plantBrain with classification, regression, and forecasting support.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ALGO8AI/plantbrain-fastml.git",
    project_urls={
        "Documentation": "https://github.com/ALGO8AI/plantbrain-fastml.git",
        "Source": "https://github.com/ALGO8AI/plantbrain-fastml.git"
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.1",
        "pandas>=1.3",
        "numpy>=1.21",
        "xgboost>=1.6",
        "lightgbm>=3.3",
        "statsmodels>=0.13",
        "matplotlib>=3.5",
        "joblib>=1.2"
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
