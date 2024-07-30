export PIP_NO_BINARY=h5py
export CC=mpicc
export HDF5_MPI="ON"
export HDF5_DIR=$(echo $(which h5copy) | awk 'BEGIN{FS=OFS="/"}{NF-=2; print}')
