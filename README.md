purf
====

## Installation
### 1. Create and activate conda envorinment:
``` bash
conda create -n purf scikit-learn=0.24.2 numpy=1.19.0 cython=0.29.21 pandas=1.3.2
conda activate purf
```

### 2. Download package:
``` bash
https://github.com/RTChou/purf.git
cd purf
``` 

### 3. Install package:
``` bash
python3 setup.py install
```

## Demo
Refer to notebooks/demo.ipynb for small data set demo. The expected output is a trained PURF model and the tested runtime is ~97 s using all threads on MacBook with 2.4 GHz 8-Core Intel Core i9 and 32 GB 2667 MHz DDR4.

## References
- Li, Chen, and Xue-Liang Hua. 2014. “Towards Positive Unlabeled Learning for Parallel Data Mining: A Random Forest Framework.” In International Conference on Advanced Data Mining and Applications, 573–87. Springer.

- De Comité, Francesco, François Denis, Rémi Gilleron, and Fabien Letouzey. 1999. “Positive and Unlabeled Examples Help Learning.” In International Conference on Algorithmic Learning Theory, 219–30. Springer.

- Denis, François, Rémi Gilleron, and Fabien Letouzey. 2005. “Learning from Positive and Unlabeled Examples.” Theor Comput Sci 348 (1): 70–83.
