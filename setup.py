# setup.py

from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description="Project created to conduct research work for master's thesis",
    author='Robert Komar',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'pydantic',
    ]
)
