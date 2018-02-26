KERNEL=_Z13search_kernel10__GPU_treeP5PointP11__GPU_pointS1_
CUBIN=kernel.sm_52.cubin
NAME=vptree
TB_SIZE=192

make clean
make DIM=7 

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.orig -t $TB_SIZE
sed -i -e 's/0.5.NEG/-0.5/g' vptree.sass.orig

make clean
make -f Makefile-reg DIM=7

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.local -t $TB_SIZE
sed -i -e 's/0.5.NEG/-0.5/g' vptree.sass.local

make clean
make -f Makefile-reg-shared DIM=7 

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass
sed -i -e 's/0.5.NEG/-0.5/g' vptree.sass
~/pyCuAsm/pyCuAsm.py $NAME.sass -t $TB_SIZE --use-local-spill
sed -i -e 's/-0.5/0.5.NEG/g' out.sass
maxas.pl -i -n out.sass $CUBIN
~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.local_shared
sed -i -e 's/0.5.NEG/-0.5/g' vptree.sass.local_shared

make clean
~/pyCuAsm/pyCuAsm.py --tuning $NAME.sass.orig --local-sass $NAME.sass.local --local-sass-shared $NAME.sass.local_shared -t $TB_SIZE

