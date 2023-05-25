# TensorFlow Optimizer of the Analytic Free Energy Model of Molecular Binding 
# This was created just to test the intricacies of the soft core functions for transfer analysis

## To employ with the most recent version of AToM
#### -l argument is Leg, specified as either 1 or another value, the latter will be leg 2
#### -s argument is for inspection of stateID (leg 1 goes from 0 to 10 and leg 2 goes from 11 to 21 for explicit)
#### -s can be 0 to 20 if analyzing hydration, for which all leg parameters are -1


### Analyzing transfer calculations:

Usage:

```
python old_sc_linear.py -o -l 1 -d data/h2o_transfer.dat
```
or ...

```
python new_sc_linear.py -o -l 1 -d data/h2o_transfer.dat
```

Inspection:

```
python old_sc_linear.py -l 1 -s 0 -d data/h2o_transfer.dat
```

or ...

```
python new_sc_linear.py -l 1 -s 0 -d data/h2o_transfer.dat
```


Restart Optimization: (The new softcore functions cause optimization to fail...)
```
python new_sc_linear.py -r -o -n 1000 -l 1 -d data/h2o_transfer.dat
```

### Analyzing hydration calculations:
#### Can ommit -l because default is -1
Usage:

```
python old_sc_linear.py -o -l -1 -d data/ammonia.dat
```

Inspection:

```
python old_sc_linear.py -l -1 -s 20 -d data/ammonia.dat
```

Restart Optimization:
```
python old_sc_linear.py -r -o -n 1000 -d data/ammonia.dat
```

(In order to use new_sc_linear.py for ammonia, change ubcore to 0 and umax to 50...)





