# Historical implementation

This directory preserves the 2021 Python implementation and its original tests and
setup instructions. They are research history, not supported 0.6 APIs. Source and
tests were moved without changing their contents. `README-original.md` preserves
the original README; root `notebooks/` and `docs/html/` retain the historical lessons.

Do not install the old requirements into the modern environment: they include
CUDA 9.1-era packages and use removed scientific Python APIs. The initial Python
3.14 collection found 15 tests and stopped with three `distutils` collection errors.
It was not a passing regression run. See `../docs/REVIEW-2021.md` for further findings.

Moving these files does not validate the numerical methods or migrate old notebooks.
Port useful ideas as separately reviewed models with independent physical tests.
