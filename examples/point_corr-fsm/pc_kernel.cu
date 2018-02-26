/* -*- mode: c -*- */

#include "pc_gpu.h"
#include "pc_kernel.h"
#include "pc_kernel_mem.h"

#define PRINT
#define PRINT_0

__global__ void init_kernel(void) {
	
}

__global__ void compute_correlation(pc_kernel_params params) {
	
	float rad;
	int pidx;
	int cur_node_index;

	extern __shared__ float tmp_shr[];
	#ifdef USE_SMEM
	__shared__ float p_coord[DIM][THREADS_PER_BLOCK];
	__shared__ gpu_node0 cur_node0[NWARPS_PER_BLOCK];
	__shared__ gpu_node1 cur_node1[NWARPS_PER_BLOCK];
	__shared__ gpu_node2 cur_node2[NWARPS_PER_BLOCK];
	__shared__ int stack[NWARPS_PER_BLOCK][128];
  //__shared__ int mask[NWARPS_PER_BLOCK][128];
	#else
	float p_coord[DIM];
	gpu_node0 cur_node0;
	gpu_node1 cur_node1;
	gpu_node2 cur_node2;
	__shared__ int SP[NWARPS_PER_BLOCK];
	__shared__ int stack[NWARPS_PER_BLOCK][64];
	//unsigned int mask[128];
	#endif

	int i, j;
	float dist, sum, boxsum, boxdist, center;
	int p_corr;
	//unsigned int cur_mask;

	bool cond, status;
    bool opt1, opt2;
	int critical;
	int id_test = 0;
	int id_complete = 0;

	#ifdef TRACK_TRAVERSALS
	int p_nodes_accessed;
	int p_nodes_truncated;
	#endif

	rad = params.rad;
	for(i = blockIdx.x*blockDim.x + threadIdx.x; i < params.npoints; i+= gridDim.x*blockDim.x) {

#define sp SP[WARP_INDEX]
		pidx = params.points[i];
		p_corr = 0; // params.tree.nodes[pidx].corr;
		
		#ifdef TRACK_TRAVERSALS
		p_nodes_accessed = 0;
		p_nodes_truncated = 0;
		#endif

		for(j = 0; j < DIM; j++) {
			p_coord[j] = params.tree.nodes0[pidx].coord[j].items.max;
		}

		STACK_INIT();
		status = 1;
		critical = 63;
		cond = 1;

		while(sp >= 1) {

			cur_node_index = STACK_TOP_NODE_INDEX;
			//cur_mask = STACK_TOP_MASK;
			CUR_NODE0 = params.tree.nodes0[cur_node_index];
			
			#ifdef TRACK_TRAVERSALS
			p_nodes_accessed++;
			#endif

			if (status == 0 && critical >= sp) {
				status = 1;
			}

			STACK_POP();

			if (status) {
				// inline call: can_correlate(...)
//				id_test ++;

//				critical = sp;
				sum = 0.0;
				boxsum = 0.0;
				for(j = 0; j < DIM; j++) {
//					center = (CUR_NODE0.coord_max[j] + CUR_NODE0.min[j]) / 2;
//					boxdist = (CUR_NODE0.coord_max[j] - CUR_NODE0.min[j]) / 2;
					center = (CUR_NODE0.coord[j].items.max + CUR_NODE0.coord[j].items.min) / 2;
					boxdist = (CUR_NODE0.coord[j].items.max - CUR_NODE0.coord[j].items.min) / 2;
					dist = p_coord[j] - center;
					sum += dist * dist;
					boxsum += boxdist * boxdist;
				}

				critical = sp;
				cond = sqrt(sum) - sqrt(boxsum) < rad;

				if(!__any(cond)) {
#ifdef TRACK_TRAVERSALS
					p_nodes_truncated++;
#endif
					continue;
				}

				if (!cond) {
					status = 0;
				} else {

					CUR_NODE1 = params.tree.nodes1[cur_node_index];

//					id_complete ++;

					if(CUR_NODE1.splitType == SPLIT_LEAF) {
						// inline call: in_radii(...)

						dist = 0.0;
						for(j = 0; j < DIM; j++) {
							dist += (p_coord[j] - CUR_NODE0.coord[j].items.max) * (p_coord[j] - CUR_NODE0.coord[j].items.max);
						}
						
						dist = sqrt(dist);
						if(dist < rad) {
							p_corr++; // = (100 * block_node_coord[0][WARP_INDEX][k]);
						}
					} else {
						CUR_NODE2 = params.tree.nodes2[cur_node_index];
						// push children
						if(CUR_NODE2.right != -1) {
							STACK_PUSH();
							STACK_TOP_NODE_INDEX = CUR_NODE2.right;
						}
				
						if(CUR_NODE2.left != -1) {
							STACK_PUSH();
							STACK_TOP_NODE_INDEX = CUR_NODE2.left;
						}
					}
				}
			}
		} 
		
		params.tree.nodes3[pidx].corr = p_corr;
		#ifdef TRACK_TRAVERSALS
		params.tree.nodes0[pidx].nodes_accessed = p_nodes_accessed;
		params.tree.nodes0[pidx].nodes_truncated = p_nodes_truncated;
		#endif
//		printf("point id = %d, max[0] = %f, test = %d, complete = %d\n", params.tree.nodes1[pidx].id, params.tree.nodes0[pidx].coord[0].item.max, id_test, id_complete);
	}
}
