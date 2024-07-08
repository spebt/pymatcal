import numpy as np
import sys
sys.path.insert(0, '..')
import pymatcal

nprocs = 90
ntasks = 187

pymatcal.get_procIds(ntasks, nprocs)
procTaskIds = pymatcal.get_procIds(ntasks, nprocs)
for pId in range(nprocs):
    if pId // 10 == 0 or pId == nprocs - 1:
        print('Process {:<5d} tId_min: {:5,} tId_max: {:5,}'.format(
            pId, procTaskIds[0, pId], procTaskIds[1, pId]))
