### Build

You'll probably need to replace compiler in Makefile (CXX).

```
make
```

### Run

```
mpirun -np 12 ./main config.ini
```

### Create plots

First create virtualenv and install packages from [requirements.txt](https://github.com/shorohml/SuperComp/blob/master/requirements.txt).

You'll need to run main with save_layers=True in config. Then
```
cd scripts 
python plot.py ../config.ini
```