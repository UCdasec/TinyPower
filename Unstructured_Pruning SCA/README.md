# Unstructured Pruning
Unstructured pruning is also called magnitude pruning. Unstructured pruning converts some of the parameters or weights with smaller magnitude into zeros.

Dense: lots of non-zero values
Sparse: lots of zeros

Unstructured pruning converts an original dense network into a sparse network. The size of the parameter matrix (or weight matrix) of the sparse network is the same as the size of parameter matrix of the original network. Sparse network has more zeros in their parameter matrix.
The unstructured pruning does not consider any relationship between the pruned weights


![image](https://github.com/UCdasec/TinyPower/assets/54579704/3d3aeefb-403d-443d-8fe7-8c47a6afde21)


# Usage

```python
cnn/train.py
```
