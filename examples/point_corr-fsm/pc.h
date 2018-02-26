#ifndef __PC_H_
#define __PC_H_

#ifndef DIM
#define DIM 7
#endif 

#ifndef RADIUS
#define RADIUS (0.32f)
#endif

#define SPLIT_LEAF (DIM)

typedef struct _kd_cell {
	int id;
	int corr;
	int splitType;
	float coord_max[DIM];
	float min[DIM];
	struct _kd_cell *left;
	struct _kd_cell *right;

	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	int nodes_truncated;
	#endif

} kd_cell;

void read_input(int argc, char *argv[]);
kd_cell * build_tree(kd_cell ** points, int split, int lb, int ub);
int kdnode_cmp(const void *a, const void *b);
void print_tree(kd_cell * root, int depth);

#endif

