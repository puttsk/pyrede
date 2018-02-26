/* -*- mode: c -*- */
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "pc.h"
#include "pc_gpu.h"
#include "pc_block.h"
#include "pc_kernel_mem.h"
#include "pc_kernel.h"
#include "hashtab.h"
#include "util_common.h"

#define CUDA_PTF_BUF_SIZE (50 * 1024 * 1024)

int npoints; // number of input points
kd_cell ** points; // points
kd_cell * root; // root of tree

unsigned long corr_sum = 0;

#ifdef TRACK_TRAVERSALS
unsigned long nodes_accessed_sum = 0;
unsigned long nodes_truncated_sum = 0;
#endif

gpu_tree * h_root; // root of host GPU tree

unsigned int sort_flag, verbose_flag, check_flag;
unsigned int ratio_flag = 0;
unsigned int warp_flag = 0;

int sort_split; /* axis component compared */
int sortidx;

TIME_INIT(overall);
TIME_INIT(read_input);
TIME_INIT(build_tree);
TIME_INIT(traverse);
TIME_INIT(gpu_tree_build);
TIME_INIT(hashtab);
TIME_INIT(init_kernel);
TIME_INIT(gpu_tree_copy_to);
TIME_INIT(kernel);
TIME_INIT(gpu_tree_copy_from);
TIME_INIT(sum_corr);
TIME_INIT(extra);

static int leaves = 0;
static int max_depth = 0;
static int number_of_nodes = 0;
void PrintTree(kd_cell * root, int depth, int id){
	if (!root) {
		return;
	}
	number_of_nodes ++;
	if (depth > max_depth) {
		max_depth = depth;
	}
	for (int i = 0; i < depth; i ++) {
		printf(" ");
	}
	printf("NODE = %d, Depth = %d, ", id, depth);
	if (root->splitType == DIM) {
		printf("Type: = LEAFNODE!\n");
		leaves ++;
	} else {
		printf("Type = %d\n", root->splitType);
	}
	if (root->left) {
		PrintTree(root->left, depth + 1, id * 2);
	}
	if (root->right) {
		PrintTree(root->right, depth + 1, id * 2 + 1);
	}
}

int main(int argc, char * argv[]) {
	int i, j;

	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (size_t) CUDA_PTF_BUF_SIZE);
	size_t size;
	cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize);
	printf("our cuda printf fifo buffer size is %d.\n", size);

	srand(0); // for quicksort

	TIME_START(overall);
	TIME_START(read_input);

	read_input(argc, argv);

	TIME_END(read_input);

	printf("configuration: sort_flag = %d, verbose_flag=%d, check_flag=%d, npoints=%d, radius=%f\n", sort_flag, verbose_flag, check_flag, npoints, RADIUS);
	
	TIME_START(build_tree);

	sortidx=0;
	root = build_tree(points, 0, 0, npoints-1);
//	PrintTree(root, 0, 1);
//	printf("Number of leaves is %d. Max depth is %d, Number of total nodes is %d.\n", leaves, max_depth, number_of_nodes);
//	exit(0);

	if(!sort_flag) {
    for(i = 0; i < npoints; i++) {
      kd_cell *tmp = points[i];
      int idx = rand() % npoints;
      points[i] = points[idx];
      points[idx] = tmp;
    }
  }
	
/*	for(i = 0; i < npoints; i++) {
		kd_cell *tmp = points[i];
		printf("%d: ", i);
		for (j = 0; j < DIM; j++) {
			printf("%f ", tmp->coord_max[j]);
		}
		printf("\n");
	}*/

	TIME_END(build_tree);
	TIME_START(traverse);

	// ** TREE BLOCKING ** //
	TIME_START(gpu_tree_build);

	h_root = build_gpu_tree(root);

	TIME_END(gpu_tree_build);

	// ** POINTER RESOLUTION **//
	TIME_START(hashtab);
	hashtab ht = hashtab_new(h_root->nnodes*3);
	for(i = 0; i < h_root->nnodes; i++) {
		if(hashtab_insert(ht, &(h_root->nodes3[i].cpu_addr), sizeof(void*), &i, sizeof(int)) != htSuccess) {
			//fprintf(stderr, "error: could not insert tree node address (%x) into hash table; could be full.", h_root->nodes[i].cpu_addr);
			exit(1);
		}
	}

	int * points_gpu;
	SAFE_MALLOC(points_gpu, sizeof(int)*h_root->nnodes);

	int gpu_tree_index;
	for(i = 0; i < npoints; i++) {
		kd_cell * key = points[i];
		if(hashtab_get(ht, &key, sizeof(void*), &gpu_tree_index, NULL) != htSuccess) {
			fprintf(stderr, "error: could not find index of tree node (%d, %x, id=%d).\n", i, points[i], points[i]->id);
			exit(1);
		}
	 
		points_gpu[i] = gpu_tree_index;
	}

	TIME_END(hashtab);

	// ** KERNEL SETUP ** //
	gpu_tree d_root;
	pc_kernel_params d_params, h_params;
	
	TIME_START(init_kernel);
	init_kernel<<<1,1>>>();
	TIME_END(init_kernel);

	//TIME_START(gpu_tree_copy_to);

	d_root.nnodes = h_root->nnodes;
	d_root.tree_depth = h_root->tree_depth;
	alloc_tree_dev(h_root, &d_root);
 
	h_params.tree = *h_root;
	d_params.tree = d_root;
	h_params.rad = d_params.rad = RADIUS;
	h_params.npoints = d_params.npoints = npoints;
	h_params.points = points_gpu;

	alloc_kernel_params_dev(&d_params);
	copy_tree_to_dev(h_root, &d_root);
	copy_kernel_params_to_dev(&h_params, &d_params);

	//TIME_END(gpu_tree_copy_to);

	// ** KERNEL ** //
	cudaError_t e;
	
	dim3 tpb(THREADS_PER_BLOCK);
	dim3 nb(NUM_THREAD_BLOCKS);
	
	printf("Kernel start!\n");

	TIME_START(kernel);
	compute_correlation<<<nb,tpb, 6144>>>(d_params);
	cudaThreadSynchronize();
	
	e=cudaGetLastError();
	if(e != cudaSuccess) {
		fprintf(stderr, "error: kernel error: %s\n", cudaGetErrorString(e));
		exit(1);
	}
	TIME_END(kernel);
	
	// ** KERNEL CLEANUP ** //
	TIME_START(gpu_tree_copy_from);
	copy_tree_to_host(h_root, &d_root);
	free_tree_dev(&d_root);

	// ** COPY BACK ** //
	for(i = 0; i < h_root->nnodes; i++) {
		kd_cell * n = h_root->nodes3[i].cpu_addr;
		if(n != 0) {
			n->corr = h_root->nodes3[i].corr;			
			#ifdef TRACK_TRAVERSALS
			n->nodes_accessed = h_root->nodes0[i].nodes_accessed;
			n->nodes_truncated = h_root->nodes0[i].nodes_truncated;
			#endif
		}
	}
	
	free_gpu_tree(h_root);

	TIME_END(gpu_tree_copy_from);
	TIME_END(traverse);

	TIME_START(sum_corr);

	for(i = 0; i < npoints; i++) {
		corr_sum += (unsigned long)points[i]->corr;
	}

	TIME_END(sum_corr);
	TIME_END(overall);

	printf("avg corr: %f\n", (float)corr_sum / npoints);
	#ifdef TRACK_TRAVERSALS
    int nwarps = 0;
	for(i = 0; i < npoints + (npoints % 32); i+=32, nwarps++) {			
		nodes_accessed_sum += (unsigned long)points[i]->nodes_accessed;
		int na = points[i]->nodes_accessed;
		int nb = points[i]->nodes_truncated;
		//printf("%d ", na);
		for(j = i + 1; j < i + 32 && j < npoints; j++) {
			//printf("%d ", points[j]->nodes_accessed);
			if(points[j]->nodes_accessed > na) {				
				na = points[j]->nodes_accessed;
				nb = points[j]->nodes_truncated;
			}
			nodes_accessed_sum += (unsigned long)points[j]->nodes_accessed;
			nodes_truncated_sum += (unsigned long)points[j]->nodes_truncated;
		}
		//printf("\n\n ***");
//		printf("nodes warp %d: %d %d\n", i/32, na, nb);
	}

    if (warp_flag) {
        int maximum = 0, all = 0;
        unsigned long long maximum_sum = 0, all_sum = 0;
        for(i = 0; i < npoints + (npoints % 32); i+=32) {
            int na = points[i]->nodes_accessed;
            maximum = na;
            all = na;

            for(j = i + 1; j < i + 32 && j < npoints; j++) {
                if(points[i]->nodes_accessed > maximum)
                    maximum = points[i]->nodes_accessed;
                all += points[i]->nodes_accessed;
            }
            printf("%d\n", maximum);
            maximum_sum += maximum;
            all_sum += all;
        } 
    }

    printf("@ sum_nodes_traversed: %ld\n", nodes_accessed_sum);
	printf("@ avg_nodes_traversed: %f\n", (float)nodes_accessed_sum / npoints);
	printf("avg nodes: %f\n", (float)nodes_accessed_sum / npoints);
	printf("avg trunc: %f\n", (float)nodes_truncated_sum / npoints);
	#endif
	
	TIME_ELAPSED_PRINT(overall, stdout);
	TIME_ELAPSED_PRINT(read_input, stdout);
	TIME_ELAPSED_PRINT(build_tree, stdout);
	TIME_ELAPSED_PRINT(hashtab, stdout);
	TIME_ELAPSED_PRINT(gpu_tree_build, stdout);
	TIME_ELAPSED_PRINT(init_kernel, stdout);
	TIME_ELAPSED_PRINT(gpu_tree_copy_to, stdout);
	TIME_ELAPSED_PRINT(kernel, stdout);
    TIME_ELAPSED_PRINT(extra, stdout);
	TIME_ELAPSED_PRINT(traverse, stdout);
	TIME_ELAPSED_PRINT(gpu_tree_copy_from, stdout);

	return 0;
}


kd_cell * build_tree(kd_cell ** points, int split, int lb, int ub) {
	int mid;
	int j;
	kd_cell *node;

	if(lb > ub)
		return 0;

	if(lb == ub) {
		return points[lb];
	} else {

		sort_split = split;
        TIME_RESTART(extra);
		qsort(&points[lb], ub - lb + 1, sizeof(kd_cell*), kdnode_cmp);
        TIME_END(extra);
		mid = (ub + lb) / 2;

		// create a new node to contains the points:
		SAFE_MALLOC(node, sizeof(kd_cell));
		node->splitType = split;

		if(mid > lb) {
			node->left = build_tree(points, (split+1) % DIM, lb, mid - 1);
			node->right = build_tree(points, (split+1) % DIM, mid, ub);
		} else {
			node->left = build_tree(points, (split+1) % DIM, lb, mid);
			node->right = build_tree(points, (split+1) % DIM, mid+1, ub);
		}

		node->corr = 0;

		for(j = 0; j < DIM; j++) {
			node->min[j] = FLT_MAX;
			node->coord_max[j] = FLT_MIN; 
		}

		if(node->left != NULL) {
			for(j = 0; j < DIM; j++) {

				if(node->coord_max[j] < node->left->coord_max[j]) {
					node->coord_max[j] = node->left->coord_max[j];
				}

				if(node->min[j] > node->left->min[j]) {
					node->min[j] = node->left->min[j];
				}
			}
		}

		if(node->right != NULL) {
			for(j = 0; j < DIM; j++) {
				if(node->coord_max[j] < node->right->coord_max[j]) {
					node->coord_max[j] = node->right->coord_max[j];
				}
				if(node->min[j] > node->right->min[j]) {
					node->min[j] = node->right->min[j];
				}
			}
		}

		return node;
	}	
}

int kdnode_cmp(const void *a, const void *b) {
	/* note: sort split must be updated before call to qsort */
	kd_cell **na = (kd_cell**)a;
	kd_cell **nb = (kd_cell**)b;
	
	if((*na)->coord_max[sort_split] < (*nb)->coord_max[sort_split]) {
		return -1;
	} else if((*na)->coord_max[sort_split] > (*nb)->coord_max[sort_split]) {
		return 1;
	} else {
		return 0;
	}
}

void read_input(int argc, char * argv[]) {
	FILE * in = stdin;
	int i, j, c, label, junk;

	printf("the number of input argument is: %d\n", argc);

	if(argc <= 2) {
		fprintf(stderr, "Usage: pc [-s] [-c] [-v] <infile> <npoints>\n");
		exit(1);
	}

	check_flag = 0;
	sort_flag = 0;
	verbose_flag = 0;

	i=0;
	while((c = getopt(argc, argv, "cvsrw")) != -1) {
		switch(c) {
		case 'c':
			check_flag = 1;
			i++;
			break;

		case 'v':
			verbose_flag = 1;
			i++;
			break;

		case 's':
			sort_flag = 1;
			i++;
			break;

		case 'r':
            ratio_flag = 1;
            break;

        case 'w':
            warp_flag = 1;
            break;

        case '?':
			fprintf(stderr, "Error: unknown option.\n");
			exit(1);
			break;

		default:
			abort();
		}
	}

	npoints = 0;
	for(i = optind; i < argc; i++) {
		switch(i - optind) {
		case 0:
			in = fopen(argv[i], "r");
			if(!in) {
				fprintf(stderr, "Error: could not open %s for reading.\n", argv[i]);
				exit(1);
			}
			break;

		case 1:
			npoints = atoi(argv[i]);
			if(npoints <= 0) {
				fprintf(stderr, "Error: invalid number of points %d.\n", npoints);
				exit(1);
			}
			break;
		}		
	}
	
	// initialize the points	
	SAFE_MALLOC(points, sizeof(kd_cell*)*npoints);

	for(i=0; i<npoints; i++) {
		SAFE_MALLOC(points[i], sizeof(kd_cell));
		points[i]->splitType = SPLIT_LEAF;
		points[i]->left = 0;
		points[i]->right = 0;
		points[i]->corr = 0;
#ifdef TRACK_TRAVERSALS		
		points[i]->nodes_accessed = 0;
		points[i]->nodes_truncated = 0;
#endif
		points[i]->id = i;

		// read coord from input
		fscanf(in, "%d", &label);
		for(j = 0; j < DIM; j++) {
			if(fscanf(in, "%f", &(points[i]->coord_max[j])) != 1) {
				fprintf(stderr, "Error: Invalid point %d\n", i);
				exit(1);
			}
			points[i]->min[j] = points[i]->coord_max[j];
		}
	}

	if(in != stdin) {
		fclose(in);
	}
	printf("%f*********\n",points[i-1]->min[j-1]);
}

