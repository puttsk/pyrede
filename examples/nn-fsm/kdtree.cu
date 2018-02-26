/* -*- mode: c -*- */
#include "nn.h"

static int sort_split;

int compare_point(const void *a, const void *b) {
	if(((struct Point *)a)->coord[sort_split] < ((struct Point *)b)->coord[sort_split]) {
		return -1;
	} else if(((struct Point *)a)->coord[sort_split] > ((struct Point *)b)->coord[sort_split]) {
		return 1;
	} else {
		return 0;
	}
}

struct KDCell * construct_tree(struct Point * points, int lb, int ub, int depth, int index) {
	struct KDCell *node = alloc_kdcell();
	int size = ub - lb + 1;
	int mid;
	int i, j;


	if (size <= MAX_POINTS_IN_CELL) {
		for (i = 0; i < size; i++) {
			node->points[i] = lb + i;
			for (j = 0; j < DIM; j++) {
				node->max[j] = max(node->max[j], points[lb + i].coord[j]);
				node->min[j] = min(node->min[j], points[lb + i].coord[j]);
				//printf("%f %f %f\n", node->max[j], node->min[j], points[lb + i].coord[j]);

			}
			//exit(0);
		}
		node->axis = DIM; // leaf node has axis of DIM
		return node;

	} else {
		sort_split = depth % DIM;
		qsort(&points[lb], ub - lb + 1, sizeof(struct Point), compare_point);
		mid = (ub + lb) / 2;

		node->axis = depth % DIM;
		node->splitval = points[mid].coord[node->axis];
		node->left = construct_tree(points, lb, mid, depth + 1, 2 * index);
		node->right = construct_tree(points, mid+1, ub, depth + 1, 2 * index + 1);

		for(j = 0; j < DIM; j++) {
			node->min[j] = min(node->left->min[j], node->right->min[j]);
			node->max[j] = max(node->left->max[j], node->right->max[j]);
			//printf("%f %f %f\n", node->max[j], node->min[j], node->left->min[j]);
		}
		return node;
	}	
}

void sort_points(struct Point * points, int lb, int ub, int depth) {
	int mid;
	if(lb >= ub)
		return;

	sort_split = depth % DIM;
	qsort(&points[lb], ub - lb + 1, sizeof(struct Point), compare_point);
	mid = (ub + lb) / 2;

	if(mid > lb) {
		sort_points(points, lb, mid - 1, depth + 1);
		sort_points(points, mid, ub, depth + 1);
	} else {
		sort_points(points, lb, mid, depth + 1);
		sort_points(points, mid+1, ub, depth + 1);
	}
}
