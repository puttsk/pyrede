/* -*- mode: c -*- */
#include <stdio.h>
#include <stdlib.h>

#include "util_common.h"
#include "nn_gpu.h"
#include "gpu_tree.h"

static gpu_point* gpu_alloc_points_host(unsigned int nnodes);
static gpu_point *gpu_alloc_points_dev(unsigned int nnodes);

static gpu_point* gpu_alloc_points_host(unsigned int npoints) {
	gpu_point *p;
	SAFE_MALLOC(p, sizeof(gpu_point)*npoints);
	return p;
}

gpu_point * gpu_transform_points(Point *points, unsigned int npoints) {
	int i, j;
	gpu_point *p = gpu_alloc_points_host(npoints);
	CHECK_PTR(p);
					
	for(i = 0; i < npoints; i++) {
		p[i].closest = points[i].closest;
		p[i].closestDist = points[i].closestDist;
		#ifdef TRACK_TRAVERSALS
		p[i].numNodesTraversed = points[i].numNodesTraversed;
		#endif
		for(j = 0; j < DIM; j++)
			p[i].coord[j] = points[i].coord[j];
	}

	return p;
}

void gpu_free_points_host(gpu_point *h_points) {
	free(h_points);
}

static gpu_point *gpu_alloc_points_dev(unsigned int npoints) {
	gpu_point *d_points;
	CUDA_SAFE_CALL(cudaMalloc(&d_points, sizeof(gpu_point)*npoints));
	return d_points;
}

void gpu_free_points_dev(gpu_point *d_points) {
	CUDA_SAFE_CALL(cudaFree(d_points));
}

gpu_point *gpu_copy_points_to_dev(gpu_point *h_points, unsigned int npoints) {
	gpu_point *d_points = gpu_alloc_points_dev(npoints);
	CUDA_SAFE_CALL(cudaMemcpy(d_points, h_points, sizeof(gpu_point)*npoints, cudaMemcpyHostToDevice));
	return d_points;
}

void gpu_copy_points_to_host(gpu_point *d_points, gpu_point *h_points, Point *points, unsigned int npoints) {
	int i;
	CUDA_SAFE_CALL(cudaMemcpy(h_points, d_points, sizeof(gpu_point)*npoints, cudaMemcpyDeviceToHost));
	for(i = 0; i < npoints; i++) {
		points[i].closest = h_points[i].closest;
		points[i].closestDist = h_points[i].closestDist;
		#ifdef TRACK_TRAVERSALS
		points[i].numNodesTraversed = h_points[i].numNodesTraversed;
		#endif
	}
}
