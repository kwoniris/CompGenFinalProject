"""
Environment configuration and setup for PDAC scGPT project.

This module sets up the environment, including:
- Random seed initialization for reproducibility
- Scanpy configuration
- Matplotlib configuration
- Directory paths
- Other global settings
"""
import os
import sys
import random
import warnings
import logging
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
import matplotlib.pyplot as plt

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
    logging.info(f"Random seeds set to {seed}")

def configure_scanpy():
    """Configure Scanpy settings"""
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Configure scanpy
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(8, 6))
    sc.settings.autoshow = False

def configure_matplotlib():
    """Configure Matplotlib settings"""
    import matplotlib
    # matplotlib.use('Agg')  # Use for saving figures without display
    plt.rcParams['figure.figsize'] = (8, 6)


# Set project root directory
ROOT_DIR = '/Users/iriskwon/Library/CloudStorage/OneDrive-JohnsHopkins/CG_Final_Proj'
RAW_DATA_DIR = '/Users/iriskwon/Library/CloudStorage/OneDrive-JohnsHopkins/CG_Final_Proj/Raw_Data'


# def get_rawdata_dir(): 
#     """Get the raw data directory for our project"""
#     return os.getenv()

# def get_results_dir():
#     """Get the results directory path"""
#     return os.getenv('PDAC_RESULTS_DIR', './results')

# def get_figures_dir():
#     """Get the figures directory path"""
#     return os.getenv('PDAC_FIGURES_DIR', './figures')

def configure_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Initialize logging
configure_logging()
