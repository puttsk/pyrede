KERNEL=_Z23nearest_neighbor_search9_gpu_treeP10gpu_point_iS1_i
CUBIN=nn.sm_52.cubin
NAME=nn
TB_SIZE=192

make clean
make 

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.orig -t $TB_SIZE
sed -i -e 's/0.5.NEG/-0.5/g' nn.sass.orig

make clean
make reg

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.local -t $TB_SIZE
sed -i -e 's/0.5.NEG/-0.5/g' nn.sass.local

make clean
make reg_shared

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass
sed -i -e 's/0.5.NEG/-0.5/g' nn.sass
~/pyCuAsm/pyCuAsm.py $NAME.sass -t $TB_SIZE --use-local-spill
sed -i -e 's/-0.5/0.5.NEG/g' out.sass
maxas.pl -i -n out.sass $CUBIN
~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.local_shared
sed -i -e 's/0.5.NEG/-0.5/g' nn.sass.local_shared

make clean
~/pyCuAsm/pyCuAsm.py --tuning $NAME.sass.orig --local-sass $NAME.sass.local --local-sass-shared $NAME.sass.local_shared -t $TB_SIZE

