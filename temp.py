import sys
print("VS Code Python:", sys.executable)

try:
    import sklearn
    print("Scikit-learn is found at:", sklearn.__file__)
except ImportError:
    print("Scikit-learn is NOT found by this Python interpreter.")
    