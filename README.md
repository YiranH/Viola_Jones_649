## Course project for ECEN 649

1. **Environment and Packages**

   The code is implemented in Python 3.6.1. 

   To run the code, you need to install 4 packages: numpy, pillow, matplotlib and pickle. 

   Numpy is used for efficient numerical calculations. 

   Pillow is used for opening and saving image files and converting images to arrays.

   Matplotlib is for drawing the top 1 feature as required in the project instruction.

   Pickle is used for storing intermediate results, which I used mainly for debugging.

   There are also several modules imported in this code: os and sys, which is used for interacting operating system to input and output data.

2. **How to run the code**

   In the same directory where the provided code and data are, you can run the code:

   $ python main.py

   (The codes are written in macOS environment. I'm not sure the instructions are right for Windows environment.)

   The results are stored in the same directory named "part1.txt", "part2.txt", "part3.txt", "part4.txt", corresponding to the 4 parts mentioned in the report.

3. **What's in this repo**

   **haar.py** is used for generating features and get intergral image.

   **adaboost.py** is used for adaboost training.

   **cascade.py** is used for the cascading system.

   **main.py** is used for running the different parts of the algorithm. 

   **helper.py** is used for calculating accuracy, false positive rate, false negative rate, etc.

   **plot.ipynb** is used for drawing plots in the report

   **image_of_round_n.png** is the output of the top 1 feature.

   **new_image_of_round_n.png** is the plotted image of **image_of_round_n.png**

   **result.png** is the plot of accuracy, fn, fp used in the report.

   **part1-4.txt** is the output of the 4 parts mentioned in the report: feature, adaboost, criteria, cascade.