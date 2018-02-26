#ifndef __PC_KERNEL_MEM_H_
#define __PC_KERNEL_MEM_H_

#include <cuda.h>
#include "pc.h"
#include "pc_gpu.h"
#include "pc_block.h"

void alloc_tree_dev(gpu_tree *h_root, gpu_tree *d_root);
void alloc_kernel_params_dev(pc_kernel_params *d_params);

void copy_tree_to_dev(gpu_tree *h_root, gpu_tree *d_root);
void copy_tree_to_host(gpu_tree *h_root, gpu_tree *d_root);
void copy_kernel_params_to_dev(pc_kernel_params *h_params, pc_kernel_params *d_params);

void free_tree_dev(gpu_tree *d_root);
void free_kernel_params_dev(pc_kernel_params *d_params);

#endif
