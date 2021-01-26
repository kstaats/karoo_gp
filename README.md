# Karoo GP

Karoo GP is an evolutionary algorithm, a genetic programming application suite 
written in Python which supports both symbolic regression and classification 
data analysis. It has been used in radio astronomy, gravitational wave detector
characterisation and synthetic supernovae detection, and a variety of other use
cases in a diversity of fields.

You need only prepare your dataset according to the User Guide. No programming
required. Karoo is multicore and GPU enabled by means of the powerful library 
TensorFlow. Karoo has three text cases built-in: Iris dataset, Kepler's law of 
planetary motion, and a maths problem you can modify to various degrees of 
challenge.

Karoo is launched from the command line with an intuitive user interface or 
with arguments for full automation from bash or another Python script. The 
output of each run is automatically archived and includes the configuraiton, a 
summary, and the full suite of GP trees saved as .csv files for your review and 
edit such that you can hand-build the starting block for your next run.

Be certain to read the User Guide for a starter's guide to Genetic Programming
and examples of all you can do with this unique body of code.

For an interesting read on scalar vs vector and CPU vs GPU performance with 
Karoo GP: https://arxiv.org/abs/1708.03157 or to learn how Karoo applied to
supernova detection at LIGO: https://arxiv.org/abs/2002.04591

Learn more at <a href="http://kstaats.github.io/karoo_gp/">kstaats.github.io/karoo_gp/</a> ...
