KERNEL=_Z24FindKeyWithDigest_KerneljjjjiiiPiPhPj
CUBIN=MD5Hash.sm_52.cubin
NAME=md5
TB_SIZE=256

make clean
make 

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.orig -t $TB_SIZE

make clean
make reg

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.local -t $TB_SIZE

make clean
make reg_shared

~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass
~/pyCuAsm/pyCuAsm.py $NAME.sass -t $TB_SIZE --use-local-spill
maxas.pl -i -n out.sass $CUBIN
~/pyCuAsm/pyCuAsm.py -e -k $KERNEL $CUBIN -o $NAME.sass.local_shared

make clean
~/pyCuAsm/pyCuAsm.py --tuning $NAME.sass.orig --local-sass $NAME.sass.local --local-sass-shared $NAME.sass.local_shared -t $TB_SIZE
