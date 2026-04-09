"""
安装配置文件
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autostat",
    version="0.1.0",
    author="AutoStat Team",
    author_email="howe_min@163.com",
    description="智能统计分析工具 - 自动识别数据类型、选择统计方法、生成分析报告",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zyhowe/AutoStat",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "networkx>=2.6.0",
        "click>=8.0.0",
        "jinja2>=3.0.0",
    ],
    extras_require={
        "web": ["streamlit>=1.0.0"],
        "mcp": ["fastmcp>=0.1.0"],
        "db": ["pyodbc>=4.0.0"],
        "excel": ["openpyxl>=3.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "all": [
            "streamlit>=1.0.0",
            "fastmcp>=0.1.0",
            "pyodbc>=4.0.0",
            "openpyxl>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autostat=autostat.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "autostat": ["templates/*.html"],
    },
    project_urls={
        "Bug Reports": "https://github.com/zyhowe/AutoStat/issues",
        "Source": "https://github.com/zyhowe/AutoStat",
    },
)