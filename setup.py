"""
Setup for kmeans module written with CAPI
"""
from setuptools import setup, Extension


setup(name="mykmeanssp",
        version="1.0",
        description="Calculate K-Means using a given initialization indices",
        ext_modules=[Extension('mykmeanssp', sources=["kmeans.c"])])

