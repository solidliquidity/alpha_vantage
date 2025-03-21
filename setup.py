from setuptools import setup, find_packages

setup(
    name="alpha_vantage_api",
    version="0.1",
    description="A Python package for tapping into Alpha Vantage's API",
    author="Luke Harding",
    url="https://github.com/solidliquidity/alpha_vantage",
    packages=find_packages(),
    python_requires='>=3.6',
)