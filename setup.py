from setuptools import setup, Extension

"""
A setup for our kmeans library
"""

setup(name="mykmeanssp",
        version="1.0",
        description="Calculate K-Means using a random initialization method",
        ext_modules=[Extension('mykmeanssp', sources=["kmeans.c"])])
