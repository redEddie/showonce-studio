"""Studio-owned live cube/bowl perception; no external source checkout is loaded."""
import os

if os.environ.get('SHOW100_CPU_AFFINITY'):
    os.sched_setaffinity(0,{int(value) for value in os.environ['SHOW100_CPU_AFFINITY'].split(',')})
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')

from .perception.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
