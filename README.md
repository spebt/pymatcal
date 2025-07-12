The folders here represent the various configurations we have tested with. Most of them contain codes to generate PPDFs, Beam Analysis codes and PPDS maps. The PPDFs are not pushed as they occupy large memory space and is not suitable for github

ORDERED CONFIGURATION
Steps 
1. Use the generate_3d_cuboids.py file to generate the cuboids based on the yaml configuration
2. Run ppdf_parallel.py to compute the system matrices. These will be saved in a folder calles sysmats
3. Run evaluate_beam_params_24rots.py to get the beam properties inside the system matrices. These beam properties will be used to compute volumes for later PPDS calculations. The properties are stacked and saved as hdf5 files for each system matrix.
4. Run calculate_beam_volume_24rots.py to get the beam volumes. These have inbuilt thresholds on the beam width which is modifiable.
5. Run calculate_ppds_maps.py to compute the ppds maps and plot it.

NOTE : While I have added the beam volumes as they were not that large in size, it is RECOMMENDED you run the sequence of steps and reproduce your own results. 

RANDOM PATTERN
Steps
1. First step is to generate our layout, to do this go in the tensor_layouts folder and run generate_randomized_scanner.py followed by transform_scanner_multiple_positions.py
2. The transform_scanner_multiple_positions.py uses the tensor generated as an argument so make sure to include that when you run it in CLI using python. This will generate all our layouts. This particular repo has 24 rotations and no translations.
3. Run ppdf_calculation_no_mpi_layouts.py to compute the system matrices. To leverage distributed processing using CPU clusters on UB CCR, use the hpc file and run via sbatch hpc on the terminal.
4. Next step are to run the extract_beams_properties.py, this will calculate the beam properties that we require into an "output" folder.  
5. Next step is to run the extact_beams_masks.py file, this will calculate the beam masks that we require into the same "output" folder.
6. After that run the analyze_extracted_properties.py to get the asci histograms.
7. After that run the asci_generation.py that will compute the asci maps and plot it
8. And at last run the calculate_and_plot_ppds_multiplexing.py/ calculate_and_plot_ppds_non_multiplexing.py based on if your requirement is for multiplexing beams or not. Usually what we observed is that for a random pattern, the non multiplexing file will not produce any result.

RANDOM PATTERN ROTS TRANS
This is just the RANDOM PATTERN with translations coupled with rotations. The sequence of steps remain exactly the same. Paralleized versions of files to compute the metrics are WIP and should be added by @Tridev.

NONMUX
This folder works with a special setup for implementing non multiplexing ppdfs. This is currently under the works so more information will be added soon. 

