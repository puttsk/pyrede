/* -*- mode: C++ -*- */

#include "vptree.h"
#include <stdio.h>
#define sp SP[WARP_INDEX]

__global__ void init_kernel(void) {
	return;
}


__global__ void search_kernel(struct __GPU_tree d_tree, struct Point *__GPU_point_Point_d, struct __GPU_point *__GPU_point_array_d, struct Point *__GPU_node_point_d) {
	
	int pidx;
	//struct Point *target;

	__shared__ int SP[NWARPS_PER_BLOCK];
	bool curr, cond, status;
    bool opt1, opt2;
	int critical;
	unsigned int vote_left;
	unsigned int vote_right;
	unsigned int num_left;
	unsigned int num_right;

	int cur_node_index;

	__shared__ int node_stack[NWARPS_PER_BLOCK][64];

	struct __GPU_Node node;
	struct __GPU_Node parent_node; // structs cached into registers

//	__shared__ struct Point target[THREADS_PER_BLOCK]; // point data cached in SMEM
	struct Point target;

	for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < d_tree.npoints; pidx += blockDim.x * gridDim.x) {
		
		target = __GPU_point_Point_d[__GPU_point_array_d[pidx].target];
		sp = 0;
		status = 1;
		critical = 63;
		cond = 1;
		node_stack[WARP_INDEX][0] = 0;
		
		while(sp >= 0) {
			
			cur_node_index = node_stack[WARP_INDEX][sp--];

			if (status == 0 && critical >= sp) {
				status = 1;
			}
            
			if (status == 1) {
				node = d_tree.nodes[cur_node_index];
				int parent_node_index = node.parent;

/*				if (pidx == 0) {
					printf("cur_index = %d\n", cur_node_index);
				}*/

#ifdef TRACK_TRAVERSALS
				target[threadIdx.x].num_nodes_traversed++;
#endif
			
				if(parent_node_index != -1) {
					parent_node = d_tree.nodes[parent_node_index];
					float upperDist = 0.0;
					int i;
					struct Point *a = &__GPU_node_point_d[parent_node.point];
					for(i = 0; i < DIM; i++) {
						float diff = (a->coord[i] - target.coord[i]);
						upperDist += (diff*diff);
					}
					upperDist = sqrt(upperDist);
				
					if(parent_node.right == cur_node_index) {
						cond = upperDist + target.tau >= parent_node.threshold;
						if(!__any(cond)) {
#ifdef TRACK_TRAVERSALS
							target[threadIdx.x].num_trunc++;
#endif
							continue;
						}
					} else if(parent_node.left == cur_node_index) {
						cond = upperDist - target.tau <= parent_node.threshold;
						if(!__any(cond)) {
#ifdef TRACK_TRAVERSALS
							target[threadIdx.x].num_trunc++;
#endif
							continue;
						}
					}
				}

				if (!cond) {
					status = 0;
					critical = sp - 1;
				} else {
					float dist = 0.0;
					int i;
					struct Point *a = &__GPU_node_point_d[node.point];
					for(i = 0; i < DIM; i++) {
						float diff = (a->coord[i] - target.coord[i]);
						dist += diff * diff;
					}
					dist = sqrt(dist);

					if(dist < target.tau) {
						target.closest_label = __GPU_node_point_d[node.point].label;
						target.tau = dist;
					}

//					int left = node.left; // cache to registers (CSE)
//					int right = node.right;
/*					if(node.left == -1 && node.right == -1) {
#ifdef TRACK_TRAVERSALS
						target[threadIdx.x].num_trunc++;
#endif
						continue;
					}*/

					opt1 = dist < node.threshold;
					opt2 = dist >= node.threshold;
					vote_left = __ballot(opt1);
					vote_right = __ballot(opt2);
					num_left = __popc(vote_left);
					num_right = __popc(vote_right);
					if(num_left > num_right) {
						if (node.right != -1) {
							node_stack[WARP_INDEX][++sp] = node.right;
						}
						if (node.left != -1) {
							node_stack[WARP_INDEX][++sp] = node.left;
						}
					} else {
						if (node.left != -1) {
							node_stack[WARP_INDEX][++sp] = node.left;
						}
						if (node.right != -1) {
							node_stack[WARP_INDEX][++sp] = node.right;
						}
					}
				}
			}
		}

		__GPU_point_Point_d[__GPU_point_array_d[pidx].target] = target;
	}
}
