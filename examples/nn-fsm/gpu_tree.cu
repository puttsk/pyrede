/* -*- mode: c -*- */
#include <stdio.h>
#include <stdlib.h>

#include "util_common.h"
#include "nn_gpu.h"
#include "gpu_tree.h"

static void gpu_alloc_tree_host(gpu_tree * h_tree);
static void gpu_init_tree_properties(gpu_tree *h_tree, KDCell *root, int depth);
static int gpu_build_tree(KDCell *root, gpu_tree *h_tree, int *index, int depth, int parent_index);
static gpu_tree* gpu_alloc_tree_dev(gpu_tree *h_tree);

gpu_tree * gpu_transform_tree(KDCell *root) {

	CHECK_PTR(root);
	
	gpu_tree *tree;
	SAFE_MALLOC(tree, sizeof(gpu_tree));

	tree->nnodes = 0;
	tree->depth = 0;

	gpu_init_tree_properties(tree, root, 0);
	gpu_alloc_tree_host(tree);
	
	int index = 0;
	gpu_build_tree(root, tree, &index, 0, NULL_NODE);

	return tree;
}

void gpu_free_tree_host(gpu_tree *h_tree) {
	CHECK_PTR(h_tree);
	free(h_tree->nodes0);	
	free(h_tree->nodes1);
  free(h_tree->nodes2);
	free(h_tree->nodes3);
}

static void gpu_alloc_tree_host(gpu_tree * h_tree) {
	SAFE_MALLOC(h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes3, sizeof(gpu_tree_node_3)*h_tree->nnodes);
}

static void gpu_init_tree_properties(gpu_tree * h_tree, KDCell * root, int depth) {

	h_tree->nnodes++;

	if(depth > h_tree->depth) 
		h_tree->depth = depth;

	if(root->left != NULL)
		gpu_init_tree_properties(h_tree, root->left, depth + 1);

	if(root->right != NULL)
		gpu_init_tree_properties(h_tree, root->right, depth + 1);
}

static int gpu_build_tree(KDCell *root, gpu_tree *h_tree, int *index, int depth, int parent_index) {
	// add node to tree
	gpu_tree_node_0 node0;
	gpu_tree_node_1 node1;
	gpu_tree_node_2 node2;
	gpu_tree_node_3 node3;
	int i;
	int my_index = *index; *index += 1;

	node0.items.axis = root->axis;
	node0.items.splitval = root->splitval;
	for(i = 0; i < DIM; i++) {
		node1.items[i].min = root->min[i];
		node1.items[i].max = root->max[i];
	}

	for(i = 0; i < MAX_POINTS_IN_CELL; i++) {
		node3.points[i] = root->points[i];
	}

	//node1.parent = parent_index;
	if(root->left != NULL)
		node2.items.left = gpu_build_tree(root->left, h_tree, index, depth + 1, my_index);
	else
		node2.items.left = NULL_NODE;
	
	if(root->right != NULL) {
		node2.items.right = gpu_build_tree(root->right, h_tree, index, depth + 1, my_index);
	} else {
		node2.items.right = NULL_NODE;
	}
	
	h_tree->nodes0[my_index] =  node0;
	h_tree->nodes1[my_index] =  node1;
	h_tree->nodes2[my_index] =  node2;
	h_tree->nodes3[my_index] =  node3;
	return my_index;
}

gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree) {

	gpu_tree * d_tree = gpu_alloc_tree_dev(h_tree);
	
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes0, h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes1, h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes2, h_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes3, h_tree->nodes3, sizeof(gpu_tree_node_3)*h_tree->nnodes, cudaMemcpyHostToDevice));
	
	return d_tree;
}

void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree) {
	// Nothing in the tree is modified to there is nothing to do here!
}

void gpu_free_tree_dev(gpu_tree *d_tree) {
	CHECK_PTR(d_tree);
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes2));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes3));
	CUDA_SAFE_CALL(cudaFree(d_tree->stk));
}

static gpu_tree* gpu_alloc_tree_dev(gpu_tree *h_tree) {
	
	CHECK_PTR(h_tree);
	
	gpu_tree * d_tree;
	SAFE_MALLOC(d_tree, sizeof(gpu_tree));
	
	// copy tree value params:
	d_tree->nnodes = h_tree->nnodes;
	d_tree->depth = h_tree->depth;

	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes));	
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes3, sizeof(gpu_tree_node_3)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->stk, sizeof(int)*h_tree->depth*NUM_THREADS_PER_BLOCK*NUM_THREAD_BLOCKS));

	return d_tree;
}
