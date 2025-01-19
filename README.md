
# `arraylib`: Multi-dimensional array library for `Python` in `C` with `numpy`-like interface

## Installation

use make to build the shared library
```bash
make
```

## Usage

```python
import arraylib as al
import math

# general example
a = (
    al.arange(1, 1 + 3 * 4 * 5 * 6)
    .reshape((3, 4, 5, 6))
    .reduce_sum(dims=(1, 2))
    .apply(lambda x: math.sqrt(x))
    .reshape((3, 6))
    .transpose((1, 0))
)

# broadcasting
b = al.ones([3, 6]) + al.ones([6]) + al.ones([4, 3, 1]) + 1.0

# stride tricks
c = al.arange(1, 11).as_strided(shape=(8, 3), stride=(1, 1)).reduce_sum(dims=[0])
```

NOTE: `omp` is supported if compiled with the appropriate flags (e.g. `-fopenmp`)
NOTE: partial SIMD (`neon`) support if compiled with the appropriate flags  (e.g. `-mfpu=neon`)