### Requirements
Code is written in Python (2.7).


### Running the method
Use the following command: 

python spgk.py filepos fileneg window_size depth

where "filepos" and "fileneg" point to the positive examples and negative examples respectively, "window_size" is the size of the sliding window, and "depth" is the maximum length of shortest paths that are taken into account


### Examples
Example command: 

python spgk.py data/subjectivity.pos data/subjectivity.neg 2 1