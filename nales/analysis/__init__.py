"""
Analysis tools for ALES data post-processing.

This subpackage provides algorithms for high-contrast imaging
analysis, including PCA-based PSF subtraction.
"""

from nales.analysis.pca import PCA, parallel_annular_PCA

__all__ = ["PCA", "parallel_annular_PCA"]
