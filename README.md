purf
====

## Installation
### 1. Create and activate conda envorinment:
``` bash
conda create -n purf scikit-learn=0.24.2 numpy=1.19.0 cython=0.29.21
conda activate purf
```

### 2. Download package:
``` bash
git clone https://gitlab.umiacs.umd.edu/rchou/purf.git
cd purf
``` 

### 3. Install package:
``` bash
python3 setup.py install
```

## References
- Li, Chen, and Xue-Liang Hua. 2014. “Towards Positive Unlabeled Learning for Parallel Data Mining: A Random Forest Framework.” In International Conference on Advanced Data Mining and Applications, 573–87. Springer.

- De Comité, Francesco, François Denis, Rémi Gilleron, and Fabien Letouzey. 1999. “Positive and Unlabeled Examples Help Learning.” In International Conference on Algorithmic Learning Theory, 219–30. Springer.

- Denis, François, Rémi Gilleron, and Fabien Letouzey. 2005. “Learning from Positive and Unlabeled Examples.” Theor Comput Sci 348 (1): 70–83.
