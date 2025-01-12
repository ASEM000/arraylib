import cffi
import os
from pathlib import Path

START = "// START"

ffi = cffi.FFI()

with open(os.path.join(Path(__file__).parent, "src", "arraylib.h")) as f:
    lines = f.read().splitlines()

ffi.cdef("\n".join(lines[lines.index("// START") : lines.index("// END")]))

# Define the C source and build options
ffi.set_source(
    "arraylib",  # Name of the generated module
    """
    #include "arraylib.h"
    """,
    include_dirs=["."],
)

lib = ffi.dlopen(os.path.join(Path(__file__).parent, "src", "arraylib.so"))
