# DeepBET for Conditional Independence Testing

This repository contains the code for implementing ""
The python version is 3.11.4

Paper link: 

## Setup
```
$ git clone https://github.com/yyang3388/DeepBET.git
$ cd DeepBET.git

# Create a virtual environment named 'myenv'
$ virtualenv myenv
$ source myenv/bin/activate

# Install dependencies
$ python3 -m pip install -r requirements.txt 
```

Example for Running the code:

```
# compute type I error for 1000 samples with multiple split with 500 simulations
$  python3 main.py \
    --test="type1error"
    --split="multiple"
    --sim_size=500
    --n_sample=1000
```
