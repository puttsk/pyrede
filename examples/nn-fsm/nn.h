#ifndef __NN_H
#define __NN_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "util_common.h"

#define MAX_POINTS_IN_CELL	1

#ifndef DIM
#define DIM 7
#endif

typedef struct Point {
	int label;
	float coord[DIM];
  int closest;
	float closestDist;
	#ifdef TRACK_TRAVERSALS
	int numNodesTraversed;
	#endif
} Point;

typedef struct KDCell {
	int axis;
	float splitval;
	float min[DIM];
	float max[DIM];
	int points[MAX_POINTS_IN_CELL];
	struct KDCell *left;
	struct KDCell *right;
} KDCell;

struct thread_args {
	int tid;
	int lb;
	int ub;	
};


void read_input(int argc, char **argv);
void read_point(FILE *in, struct Point *p);

void nearest_neighbor_search(struct Point *point, struct KDCell *root);
int can_correlate(struct Point * point, struct KDCell * cell, float rad);
void update_closest(struct Point *point, int candidate_index);

struct Point* alloc_points(int n);
struct KDCell *alloc_kdcell();

struct KDCell * construct_tree(struct Point *points, int start_idx, int end_idx, int depth, int index);
void sort_points(struct Point *points, int start_idx, int end_idx, int depth);
int compare_point(const void *a, const void *b);

#endif
