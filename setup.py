"""
QuantLab - LLM Quantization Experimentation Platform

Setup script for pip installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quantlab",
    version="0.1.0",
    author="QuantLab Team",
    description="LLM Quantization Experimentation Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=" https://github.com/hemantsingh443/quantization-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "psutil>=5.9.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.18.0",
            "pandas>=2.0.0",
        ],
        "gpu": [
            "pynvml>=11.5.0",
        ],
        "gptq": [
            "auto-gptq>=0.5.0",
        ],
        "awq": [
            "autoawq>=0.1.0",
        ],
        "eval": [
            "lm-eval>=0.4.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "matplotlib>=3.7.0",
        ],
        "all": [
            "streamlit>=1.28.0",
            "plotly>=5.18.0",
            "pandas>=2.0.0",
            "pynvml>=11.5.0",
            "matplotlib>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantlab=quantlab.dashboard.cli:main",
        ],
    },
)
