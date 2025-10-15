"""
Setup script for SecureAI Personal Assistant
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="secure-ai-assistant",
    version="0.1.0",
    author="SecureAI Team",
    description="A privacy-first personal assistant with advanced security features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/secure-ai-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "ollama>=0.1.0",
        "google-api-python-client>=2.0.0",
        "google-auth>=2.0.0",
        "google-auth-oauthlib>=1.0.0",
        "google-auth-httplib2>=0.2.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "pytesseract>=0.3.10",
        "shap>=0.42.0",
        "lime>=0.2.0",
        "watchdog>=3.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "sqlite-utils>=3.30.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "click>=8.0.0",
        "colorama>=0.4.6",
        "tqdm>=4.65.0",
        "pyyaml>=6.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "secure-assistant=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md", "*.txt"],
    },
)
