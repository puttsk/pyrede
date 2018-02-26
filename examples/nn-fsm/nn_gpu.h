#ifndef __NN_GPU_H
#define __NN_GPU_H

#include "nn.h"
#include "gpu_tree.h"

#define WARP_SIZE (32)

#define NUM_THREAD_BLOCKS (1024)
#define NUM_THREADS_PER_BLOCK (WARP_SIZE*6)
#define NUM_THREADS_PER_GRID (NUM_THREAD_BLOCKS * NUM_THREADS_PER_BLOCK)

#define NUM_WARPS_PER_BLOCK (NUM_THREADS_PER_BLOCK / WARP_SIZE)
#define NUM_WARPS_PER_GRID (NUM_THREADS_PER_GRID / WARP_SIZE)

#define WARP_IDX (threadIdx.x >> 5)
#define GLOBAL_WARP_IDX (WARP_IDX + (blockIdx.x*NUM_WARPS_PER_BLOCK))
#define THREAD_IDX_IN_WARP (threadIdx.x - (WARP_SIZE * WARP_IDX))
#define IS_FIRST_THREAD_IN_WARP (threadIdx.x == (WARP_IDX * WARP_SIZE))

// top levelfunctions
gpu_tree * gpu_transform_tree(KDCell *root);
gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree);
void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree);
void gpu_free_tree_dev(gpu_tree *d_tree);
void gpu_free_tree_host(gpu_tree *h_tree);
void gpu_print_tree_host(gpu_tree *h_tree);

gpu_point * gpu_transform_points(Point *points, unsigned int npoints);
gpu_point * gpu_copy_points_to_dev(gpu_point *h_points, unsigned int npoints);
void gpu_copy_points_to_host(gpu_point *d_points, gpu_point *h_points, Point *points, unsigned int npoints);
void gpu_free_points_dev(gpu_point *d_points);
void gpu_free_points_host(gpu_point *h_points);

__global__ void init_kernel(void);
__global__ void nearest_neighbor_search (gpu_tree gpu_tree, gpu_point *d_training_points, int n_training_points, gpu_point *d_search_points, int n_search_points);
#endif
