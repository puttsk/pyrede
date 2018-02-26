make clean
make 

~/pyCuAsm/pyCuAsm.py -e -k _Z17cuda_compute_fluxiPiPfS0_S0_S0_6float3S1_S1_S1_ euler3d.sm_52.cubin -o euler3d.sass.orig -t 192

make clean
make reg

~/pyCuAsm/pyCuAsm.py -e -k _Z17cuda_compute_fluxiPiPfS0_S0_S0_6float3S1_S1_S1_ euler3d.sm_52.cubin -o euler3d.sass.local -t 192

make clean
make reg_shared

~/pyCuAsm/pyCuAsm.py -e -k _Z17cuda_compute_fluxiPiPfS0_S0_S0_6float3S1_S1_S1_ euler3d.sm_52.cubin -o euler3d.sass
~/pyCuAsm/pyCuAsm.py euler3d.sass -t 192 --use-local-spill
maxas.pl -i -n out.sass euler3d.sm_52.cubin
~/pyCuAsm/pyCuAsm.py -e -k _Z17cuda_compute_fluxiPiPfS0_S0_S0_6float3S1_S1_S1_ euler3d.sm_52.cubin -o euler3d.sass.local_shared

make clean
~/pyCuAsm/pyCuAsm.py --tuning euler3d.sass.orig --local-sass euler3d.sass.local --local-sass-shared euler3d.sass.local_shared -t 192

