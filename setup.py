from setuptools import setup, find_packages

setup(
    name="document-impact-analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.23.0",
        "transformers>=4.0.0",
        "PyPDF2>=2.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "wordcloud>=1.8.0",
    ],
    extras_require={
        "full": [
            "langchain>=0.0.200",
            "langchain-huggingface>=0.0.1",
            "langchain-chroma>=0.0.1",
            "pdfplumber>=0.7.0",
            "torch>=1.9.0",
        ],
    },
    author="Rocky Auer",
    author_email="rocky2397@icloud.com",
    description="A tool for analyzing the impact of specific topics in corporate documents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rocky2397/document-impact-analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)