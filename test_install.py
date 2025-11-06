# test_install.py
"""Test if all required packages are installed"""

print("Testing PRISM installation...")
print("=" * 60)

# Check Python version first
import sys
print(f"Python version: {sys.version}")
print()

# Test each package
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print(f"  (Running on CPU - this is fine for testing)")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    print(f"✓ Sentence Transformers {sentence_transformers.__version__}")
except ImportError as e:
    print(f"✗ Sentence Transformers: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn: {e}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")

try:
    import seaborn
    print(f"✓ Seaborn {seaborn.__version__}")
except ImportError as e:
    print(f"✗ Seaborn: {e}")

try:
    import datasets
    print(f"✓ Datasets {datasets.__version__}")
except ImportError as e:
    print(f"✗ Datasets: {e}")

try:
    import tqdm
    print(f"✓ tqdm {tqdm.__version__}")
except ImportError as e:
    print(f"✗ tqdm: {e}")

print("=" * 60)
print("Installation test complete!")
print()

# Count successes
print("Summary:")
print("If you see ✓ for all packages above, you're ready to go!")
print("If you see any ✗, those packages need to be installed.")
