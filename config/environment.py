# config/environment.py
"""
Environment configuration and setup for PDAC scGPT project
"""

import os
import sys
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import scanpy as sc

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed}")

def configure_environment():
    """Configure scanpy, matplotlib, and other settings"""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Configure scanpy
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(6, 4))
    sc.settings.autoshow = False
    
    # Set matplotlib backend for non-interactive use
    import matplotlib
    # matplotlib.use('Agg')  # Use for saving figures without display
    
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (8, 6)