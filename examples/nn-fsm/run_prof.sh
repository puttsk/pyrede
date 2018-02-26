#make clean;
#make DIM=3;
#./nn ../../../input/pc/geocity.txt 200000

make clean;
make DIM=7;
./nn -s ../../../input/pc/covtype.7d 200000
#./nn -s ../../../input/pc/covtype.7d 1000 > my.out
#vim my.out
./nn -s ../../../input/pc/mnist.7d 200000

./nn -s ../../../input/pc/random.7d 200000

