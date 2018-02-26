#ifndef __PC_BLOCK_H
#define __PC_BLOCK_H

#include <cuda.h>
#include "util_common.h"
#include "pc_gpu.h"

gpu_tree * build_gpu_tree(kd_cell * c_root);
int block_gpu_tree(kd_cell * c_node, gpu_tree * h_root, int * index, int depth);
void block_tree_info(gpu_tree * gpu_root, kd_cell * root, int depth);

void free_gpu_tree(gpu_tree * root);
void print_gpu_tree(gpu_tree * root, int id, int depth);

#endif
