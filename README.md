
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
    .sum(dims=(1, 2))
    .apply(lambda x: math.sqrt(x))
    .reshape((3, 6))
    .transpose((1, 0))
)

# broadcasting
b = al.ones([3, 6]) + al.ones([6]) + al.ones([4, 3, 1]) + 1.0
```