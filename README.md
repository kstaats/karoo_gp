# Karoo GP

Karoo GP is an evolutionary algorithm, a genetic programming application suite written in Python which supports both symbolic regression and classification data analysis. It is ready to work with your datasets, is multicore and GPU enabled by means of the powerful library TensorFlow. The packge includes a Desktop application with an intuitive user interface, and a Server application which supports fully scripted runs. Output is automatically archived.

Batteries included. No programming required.

Learn more at [kstaats.github.io/karoo_gp](http://kstaats.github.io/karoo_gp)

## Dependencies:

- python (3.6.2)
- pip (9.0.1)
- matplotlib (2.0.2)
- numpy (1.13.1)
- scikit-learn (0.19.0)
- scipy (0.19.1)
- sklearn (0.0)
- sympy (1.1.1)
- tensorflow (1.2.1)

## Using Karoo GP

[Before you do anything read the Karoo GP User Guide](https://github.com/kstaats/karoo_gp/blob/master/)

### Installation (With Anaconda)

```
create -n tensorflow
source activate tensorflow
pip install numpy sympy tensorflow scipy scikit-learn matplotlib sklearn
```

### Starting Karoo GP Interface

```
 python karoo_gp_main.py
```

### Using your own data

```
 python karoo_gp_main.py /path/to/data.csv
```

### Running Server

```
 python karoo_gp_server.py
```
