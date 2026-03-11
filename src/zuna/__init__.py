"""
Zuna: a 380M-parameter masked diffusion autoencoder EEG Foundation Model trained to reconstruct, denoise, and upsample scalp-EEG signals.  

Main functions:
    zuna.preprocessing()          - .fif → .pt (resample, filter, epoch, normalize)
    zuna.inference()              - .pt → .pt (model reconstruction)
    zuna.pt_to_fif()              - .pt → .fif (denormalize, concatenate)
    zuna.compare_plot_pipeline()  - Generate comparison plots
    zuna.extract_features()       - .pt → pooled/token embeddings

See tutorials/run_zuna_pipeline.py for a complete working example.
Use help(zuna.preprocessing) etc. for detailed documentation.
"""

__version__ = "0.1.1"

from .preprocessing.batch import preprocessing
from .pipeline import inference, pt_to_fif, extract_features
from .visualization.compare import compare_plot_pipeline

__all__ = [
    'preprocessing',
    'inference',
    'extract_features',
    'pt_to_fif',
    'compare_plot_pipeline',
]
