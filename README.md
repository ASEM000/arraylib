
In-progress building minimial `jax`-like numerical transformation in python with `numpy`-like library in C.

use `make` to build the shared libray.


```python
import arraylib as al
def fn(a, b):
    return al.exp(a**b + 1.0)

# primal, tangent pair
a = al.JVPNDArray(al.array([2.0]), al.array([1.0])) 
b = al.JVPNDArray(al.array([3.0]), al.array([0.0]))
out = fn(a, b)
primal, tanget = out.primal, out.tangent
print(primal)
print(tanget)
#[8103.083984375]
#[97237.0078125]
```