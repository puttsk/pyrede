KERNEL=_Z19compute_correlation17_pc_kernel_params
CUBIN=pc_kernel.sm_52.cubin
NAME=pc
TB_SIZE=192

make clean
make DIM=7 RADIUS=0.32f

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.orig -t $TB_SIZE
sed -i -e 's/0.5.NEG/-0.5/g' pc.sass.orig

make clean
make -f Makefile-reg DIM=7 RADIUS=0.32f

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.local -t $TB_SIZE
sed -i -e 's/0.5.NEG/-0.5/g' pc.sass.local

make clean
make -f Makefile-reg-shared DIM=7 RADIUS=0.32f

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass
sed -i -e 's/0.5.NEG/-0.5/g' pc.sass
~/pyCuAsm/pyCuAsm.py $NAME.sass -t $TB_SIZE --use-local-spill
sed -i -e 's/-0.5/0.5.NEG/g' out.sass
maxas.pl -i -n out.sass $CUBIN
~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.local_shared
sed -i -e 's/0.5.NEG/-0.5/g' pc.sass.local_shared

make clean
~/pyCuAsm/pyCuAsm.py --tuning $NAME.sass.orig --local-sass $NAME.sass.local --local-sass-shared $NAME.sass.local_shared -t $TB_SIZE

