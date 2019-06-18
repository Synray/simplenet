# simplenet
A simple feedforward neural network written in under 500 lines of C.

With one hidden layer of 50 neurons, the network achieves a maximum test accuracy of 97.5% on the MNIST data set.

## Compiling and Running

The program is set up to train on the MNIST database of handwritten digits, so you'll need to download the database and place the files in the same directory as the executable.

### Linux

Just run the following commands:

```bash
make
# Download the MNIST database
wget http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
# Unzip the database files
gunzip {train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
# Run the program
./snet
```

### Windows

The code was written on Linux, and since Windows does not support C very well, the code might not compile without a few changes.
Simplenet uses some C11 features and GCC extensions. There are a few `#ifdef`'s to turn them off, but so far the code has only been compiled and tested on Linux.

After compiling, you'll need to download and unzip the four files from [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/).
