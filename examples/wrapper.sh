echo $(date)
mpirun -n 4 python get_all_ppdfs_mpi.py test_20240722_182815.yml
echo $(date)
