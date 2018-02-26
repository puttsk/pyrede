/* -*- mode: c -*- */

#include <stdio.h>

#include "pc_block.h"


gpu_tree * build_gpu_tree(kd_cell * c_root) {

	int index;
	
	gpu_tree * h_tree;
	SAFE_MALLOC(h_tree, sizeof(gpu_tree));
	
	// get information from the cpu tree
	h_tree->nnodes = 0;
	h_tree->tree_depth = 0;
	block_tree_info(h_tree, c_root, 1);

	// allocate the tree	
	SAFE_MALLOC(h_tree->nodes0, sizeof(gpu_node0)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes1, sizeof(gpu_node1)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes2, sizeof(gpu_node2)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes3, sizeof(gpu_node3)*h_tree->nnodes);

	index = 0;
	block_gpu_tree(c_root, h_tree, &index, 0);

//	print_gpu_tree(h_tree, 0, 0);
//
//	printf("\n\n");
 
	return h_tree;
}

int block_gpu_tree(kd_cell * c_node, gpu_tree * root, int * index, int depth) {

	int i;
	int my_index = -1;

	// Save the current index as ours and go to next free position
	my_index = *index;
	*index = *index + 1; 

	// copy the node data	
	root->nodes3[my_index].corr = 0;
	root->nodes1[my_index].splitType = c_node->splitType;
	root->nodes1[my_index].id = c_node->id;
	for(i = 0; i < DIM; i++) {
//		root->nodes0[my_index].coord_max[i] = c_node->coord_max[i];
//		root->nodes0[my_index].min[i] = c_node->min[i];
		root->nodes0[my_index].coord[i].items.max = c_node->coord_max[i];
		root->nodes0[my_index].coord[i].items.min = c_node->min[i];
	}

	root->nodes3[my_index].cpu_addr = c_node;
	
	if(c_node->left != NULL) 
		root->nodes2[my_index].left = block_gpu_tree(c_node->left, root, index, depth+1);
	else
		root->nodes2[my_index].left = -1;

	if(c_node->right != NULL)
		root->nodes2[my_index].right = block_gpu_tree(c_node->right, root, index, depth+1);
	else
		root->nodes2[my_index].right = -1;

	#ifdef TRACK_TRAVERSALS
	root->nodes0[my_index].nodes_accessed = 0;
	#endif

	// return the node index
	return my_index;
}

void block_tree_info(gpu_tree * h_root, kd_cell * c_root, int depth) {

  // update maximum depth
	if(depth > h_root->tree_depth)
		h_root->tree_depth = depth;
 
	// update number of nodes
	h_root->nnodes++;

	// goto children
	if(c_root->left != NULL) {
		block_tree_info(h_root, c_root->left, depth+1);
	}

	if(c_root->right != NULL) {
		block_tree_info(h_root, c_root->right, depth+1);
	}
}

void free_gpu_tree(gpu_tree * h_root) {
	free(h_root->nodes0);
	free(h_root->nodes1);
	free(h_root->nodes2);
	free(h_root->nodes3);
	free(h_root);
}

void print_gpu_tree(gpu_tree * h_root, int id, int depth) {
	if (id == -1) {
		return;
	}
	int left = h_root->nodes2[id].left;
	int right = h_root->nodes2[id].right;
	for (int i = 0; i < depth; i ++) {
		printf(" ");
	}
	printf("Node %d, Depth %d ", id, depth);
	if (left != -1) {
		printf("has a left child as node %d\n", left);
		print_gpu_tree(h_root, left, depth + 1);
	}

	if (right != -1) {
		for (int i = 0; i < depth; i ++) {
			printf(" ");
		}
		printf("Node %d, Depth %d ", id, depth);
		printf("has a right child as node %d\n", right);
		print_gpu_tree(h_root, right, depth + 1);
	}
	if (left == -1 && right == -1) {
		printf("has no child!\n");
	}
}


