/* -*- mode: c -*- */

#include <stdio.h>

#include "pc.h"
#include "pc_kernel_mem.h"
#include "pc_gpu.h"

void alloc_tree_dev(gpu_tree *h_root, gpu_tree *d_root) {
	CUDA_SAFE_CALL(cudaMalloc(&(d_root->nodes0), sizeof(gpu_node0)*h_root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_root->nodes1), sizeof(gpu_node1)*h_root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_root->nodes2), sizeof(gpu_node2)*h_root->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&(d_root->nodes3), sizeof(gpu_node3)*h_root->nnodes));
}

void alloc_kernel_params_dev(pc_kernel_params *d_params) {
	CUDA_SAFE_CALL(cudaMalloc(&(d_params->points), sizeof(int) * d_params->npoints));
}

void copy_tree_to_dev(gpu_tree *h_root, gpu_tree *d_root) {
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes0, h_root->nodes0, sizeof(gpu_node0)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes1, h_root->nodes1, sizeof(gpu_node1)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes2, h_root->nodes2, sizeof(gpu_node2)*h_root->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_root->nodes3, h_root->nodes3, sizeof(gpu_node3)*h_root->nnodes, cudaMemcpyHostToDevice));
}

void copy_tree_to_host(gpu_tree *h_root, gpu_tree *d_root) {	
	CUDA_SAFE_CALL(cudaMemcpy(h_root->nodes3, d_root->nodes3, sizeof(gpu_node3)*h_root->nnodes, cudaMemcpyDeviceToHost));
	#ifdef TRACK_TRAVERSALS
	CUDA_SAFE_CALL(cudaMemcpy(h_root->nodes0, d_root->nodes0, sizeof(gpu_node0)*h_root->nnodes, cudaMemcpyDeviceToHost));
	#endif
}

void copy_kernel_params_to_dev(pc_kernel_params *h_params, pc_kernel_params *d_params) {	
	CUDA_SAFE_CALL(cudaMemcpy(d_params->points, h_params->points, sizeof(int)*d_params->npoints, cudaMemcpyHostToDevice));
}

void free_tree_dev(gpu_tree *d_root) {	
	CUDA_SAFE_CALL(cudaFree(d_root->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes2));
	CUDA_SAFE_CALL(cudaFree(d_root->nodes3));
}

void free_kernel_params_dev(pc_kernel_params *d_params) {	
	CUDA_SAFE_CALL(cudaFree(d_params->points));
}



