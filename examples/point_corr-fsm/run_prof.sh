make clean;
make DIM=7 RADIUS=0.32f;
#./pc -s ../../../input/pc/covtype.7d 200000 > my.out
#vim my.out


./pc -s covtype.7d 200000
#./pc ../../../input/pc/covtype.7d 200000
#./pc -s ../../../input/pc/mnist.7d 200000
#./pc ../../../input/pc/mnist.7d 200000
#./pc -s ../../../input/pc/random.7d 200000
#./pc ../../../input/pc/random.7d 200000

#make clean;
#make DIM=3 RADIUS=0.1f;

#./pc -s ../../../input/pc/geocity.txt 200000
#./pc ../../../input/pc/geocity.txt 200000

