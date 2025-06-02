Ensure that all dependencies are installed by navigating to the Codebase directory and running
pip install requirements.txt. Then, to run a specific algorithm, follow this process:
1. If you wish to add your own TSP Instance to run with one of our algorithm’s, ensure it is
an .xlsx file that corresponds with our valid structure and add it to the directory called
TSP Utilities/Test Inputs/TSP Instances. The valid structure is detailed in the Design
section of this report (specifically section 5.2.3 of the report).
2. Navigate into that algorithm’s directory.
3. Make a call at the bottom of the script to the algorithm’s relevant run functions. Many
examples can be found commented out at the bottom of each script.
4. Run python .py, replacing the gap with the algorithm’s filename. For example, to run
NN, run python nearest neighbour.py.
5. If you wish graph to be displayed on running the algorithm, ensure that the display route
parameter of the algorithm’s run function is set to True.

All .xlsx instances for invalid testing can be found in the 'Invalid TSP Inputs' folder.