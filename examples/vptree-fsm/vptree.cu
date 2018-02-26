/* -*- mode: c++ -*- */
// A VP-Tree implementation, by Steve Hanov. (steve.hanov@gmail.com)
// Released to the Public Domain
// Based on "Data Structures and Algorithms for Nearest Neighbor Search" by Peter N. Yianilos
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <float.h>
#include <getopt.h>
#include "util_common.h"
#include "vptree.h"
#include "ptrtab.h"

struct Point *__GPU_point_Point_d;
struct Point *__GPU_point_Point_h;
struct __GPU_point *__GPU_point_array_d;
struct __GPU_point *__GPU_point_array_h;
struct Point *__GPU_node_point_d;
struct Point *__GPU_node_point_h;

struct Point **_items;
int npoints;
int verbose_flag;
int check_flag;
int sort_flag;
int ratio_flag = 0;
int warp_flag = 0;
int nthreads;
struct Node *_root;

TIME_INIT(search_kernel);
TIME_INIT(kernel);
TIME_INIT(sort);
TIME_INIT(init_kernel);
TIME_INIT(search);

static int leaves = 0;
static int max_depth = 0;
static int number_of_nodes = 0;
/*void PrintTree(Node * root, int depth, int id){
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
	if (root->left == NULL && root->right == NULL) {
		printf("Type: = LEAFNODE!\n");
		leaves ++;
	} else {
		printf("Type = non\n");
	}
	if (root->left) {
		PrintTree(root->left, depth + 1, id * 2);
	}
	if (root->right) {
		PrintTree(root->right, depth + 1, id * 2 + 1);
	}
}*/

int compare_point(const void *p1, const void *p2) {
  const struct Point *pp1 =  *((const struct Point **)p1);
  const struct Point *pp2 =  *((const struct Point **)p2);
  if ((pp1 -> vantage_dist) < (pp2 -> vantage_dist)) 
    return -1;
  else 
    return 1;
}

struct Node* buildFromPoints(struct Point **_items, int lower, int upper ) {
	int i, j;
	struct Point *ptmp;
	
	#ifdef TRACK_TRAVERSALS
	static int node_id = 0;
	#endif

	if ( upper == lower ) {
		return NULL;
	}

	struct Node* node = NULL;
	SAFE_MALLOC(node, sizeof(struct Node));

	node->point = _items[lower];
	node->left = NULL;
	node->right = NULL;
	node->parent = NULL;
	node->threshold = 0.0;
	#ifdef TRACK_TRAVERSALS
	node->id = node_id++;
	#endif

	if ( upper - lower > 1 ) {

		// choose an arbitrary point and move it to the start
		// This is one of several ways to find the best candidate VP
		i = lower; //(int)((float)rand() / RAND_MAX * (upper - lower - 1) ) + lower;
		ptmp = _items[lower];
		_items[lower] = _items[i];
		_items[i] = ptmp;

		int median = ( upper + lower ) / 2;

		// partitian around the median distance		
		for(i = lower + 1; i < upper; i++) {
			_items[i]->vantage_dist = mydistance(_items[lower], _items[i]);
		}

		qsort(&_items[lower + 1], upper - lower - 1, sizeof(struct Point *), compare_point);

		// what was the median?
		node->threshold = mydistance( _items[lower], _items[median]);

		node->point = _items[lower];
		node->left = buildFromPoints(_items, lower + 1, median );

		if (node->left != NULL) 
			node->left->parent = node;
		
		node->right = buildFromPoints(_items, median, upper );

		if (node->right != NULL) 
			node->right->parent = node;
	}

	return node;
}

// to avoid conflict with std::distance :/
float mydistance(struct Point *a,struct Point *b) {
  float d = 0.0;
  int i;
  for (i = 0; i < DIM; i++) {
    float diff = ((a -> coord)[i] - (b -> coord)[i]);
    d += (diff * diff);
  }
  return (sqrt(d));
}

void *search_entry(void *args)
{
  targs *ta = (targs *)args;
  struct Point *target;
  int i;
  
	TIME_START(search_kernel);

  __GPU_point_array_h = ((struct __GPU_point *)(malloc(sizeof(struct __GPU_point ) * ((ta -> ub) - (ta -> lb)))));
  if (__GPU_point_array_h == 0) {
    fprintf(stderr,"error [file=%s line=%d]: %s is NULL!","transformation",0,"__GPU_point_array_h");
    abort();
  }
  __GPU_point_Point_h = ((struct Point *)(malloc(sizeof(struct Point ) * ((ta -> ub) - (ta -> lb)))));
  if (__GPU_point_Point_h == 0) {
    fprintf(stderr,"error [file=%s line=%d]: %s is NULL!","transformation",0,"__GPU_point_Point_h");
    abort();
  }
  __GPU_node_point_h = ((struct Point *)(malloc(sizeof(struct Point ) * ((ta -> ub) - (ta -> lb)))));
  if (__GPU_node_point_h == 0) {
    fprintf(stderr,"error [file=%s line=%d]: %s is NULL!","transformation",0,"__GPU_node_point_h");
    abort();
  }

  for (i = (ta -> lb); i < (ta -> ub); i++) {
    target = (ta -> searchpoints)[i];
    target -> tau = 3.40282347e+38F;
    memcpy(__GPU_point_Point_h + i,target,sizeof(struct Point ) * 1);
    __GPU_point_array_h[i].target = i;
  }

  struct __GPU_tree __the_tree_h;
  __the_tree_h = __GPU_buildTree(_root,(ta -> ub) - (ta -> lb));

  struct __GPU_tree __the_tree_d;
  __the_tree_d = __GPU_allocDeviceTree(__the_tree_h);
  __GPU_memcpyTreeToDev(__the_tree_h,__the_tree_d);
  
	dim3 blocks(NUM_THREAD_BLOCKS);
	dim3 tpb(THREADS_PER_BLOCK);

	TIME_START(kernel);
	search_kernel <<<blocks, tpb, 4096>>> (__the_tree_d, __GPU_point_Point_d, __GPU_point_array_d, __GPU_node_point_d);
	cudaThreadSynchronize();		
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) {
		fprintf(stderr, "Error: search_kernel failed with error: %s\n", cudaGetErrorString(e));
		exit(1);
	}
	TIME_END(kernel);
	TIME_ELAPSED_PRINT(kernel, stdout);
    TIME_ELAPSED_PRINT(sort, stdout);
  
  __GPU_memcpyTreeToHost(__the_tree_h, __the_tree_d);

	for (i = (ta -> lb); i < (ta -> ub); i++) {
		target = (ta -> searchpoints)[i];
		memcpy(target, __GPU_point_Point_h + i,sizeof(struct Point ) * 1);
  }

	__GPU_freeDeviceTree(__the_tree_d);

	TIME_END(search_kernel);
	TIME_ELAPSED_PRINT(search_kernel, stdout);

  pthread_exit(0);
  return 0;
}

struct Point *read_point(FILE *in) {
#ifdef TRACK_TRAVERSALS
	static int id = 0;
#endif

	struct Point *p;
	SAFE_MALLOC(p, sizeof(struct Point));
		
	if(fscanf(in, "%d", &p->label) != 1) {
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}
	int j;
	for(j = 0; j < DIM; j++) {
		if(fscanf(in, "%f", &p->coord[j]) != 1) {
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}
	}

#ifdef TRACK_TRAVERSALS
	p->num_nodes_traversed = 0;
	p->num_trunc = 0;
	p->id = id++;
#endif

	return p;
}

struct Point *gen_point() {
#ifdef TRACK_TRAVERSALS
	static int id = 0;
#endif

	struct Point *p;
	SAFE_MALLOC(p, sizeof(struct Point));
	int j;
	p->label=0;
	for (j = 0; j < DIM; j++) {
		p->coord[j] = (float)rand() / RAND_MAX;
	}

#ifdef TRACK_TRAVERSALS
	p->num_nodes_traversed = 0;
	p->num_trunc = 0;
	p->id = id++;
#endif

	return p;
}

void read_input(int argc, char **argv, struct Point*** p_points, struct Point*** p_searchpoints) {
	
	int i, c;
	struct Point **points;
	struct Point **searchpoints;

	check_flag = 0;
	sort_flag = 0;
	verbose_flag = 0;
	nthreads = 1;
	i=0;
	while((c = getopt(argc, argv, "cvt:srw")) != -1) {
		switch(c) {
		case 'c':
			check_flag = 1;
			i++;
			break;

		case 'v':
			verbose_flag = 1;
			i++;
			break;

		case 't':
			nthreads = atoi(optarg);
			if(nthreads <= 0) {
				fprintf(stderr, "Error: invalid number of threads.\n");
				exit(1);
			}
			i+=2;
			break;

		case 's':
			sort_flag = 1;
			i++;
			break;

        case 'r':
            ratio_flag = 1;
            i ++;
            break;

        case 'w':
            warp_flag = 1;
            i ++;
            break;

		case '?':
			fprintf(stderr, "Error: unknown option.\n");
			exit(1);
			break;

		default:
			abort();
		}
	}
 
	if(argc - i < 2 || argc - i > 3) {
		fprintf(stderr, "usage: vptree [-c] [-v] [-t <nthreads>] [-s] <npoints> [input_file]\n");
		exit(1);
	}

	char *input_file = NULL;
	for(i = optind; i < argc; i++) {
		switch(i - optind) {
		case 0:
			npoints = atoi(argv[i]);
			if(npoints <= 0) {
				fprintf(stderr, "Invalid number of points.\n");
				exit(1);
			}
			break;

		case 1:
			input_file = argv[i];
			break;
		}
	}

	printf("Configuration: sort_flag = %d, verbose_flag = %d, nthreads=%d, DIM = %d, npoints = %d, input_file=%s\n", sort_flag, verbose_flag, nthreads, DIM, npoints, input_file);

	// Allocate the point and search point arrays
	SAFE_CALLOC(points, npoints, sizeof(struct Point*));
	SAFE_CALLOC(searchpoints, npoints, sizeof(struct Point*));

	if (input_file != NULL) {
		FILE *in = fopen(input_file, "r");
		if( in == NULL) {
			fprintf(stderr, "Could not open %s\n", input_file);
			exit(1);
		}

		for (i = 0; i < npoints; i++) {
			points[i] = read_point(in);
		}

		for (i = 0; i < npoints; i++) {
			searchpoints[i] = read_point(in);
		}

		fclose(in);
	} else {
		for (i = 0; i < npoints; i++) {
			points[i] = gen_point();			
		}
		for (i = 0; i < npoints; i++) {
			searchpoints[i] = gen_point();
		}
	}
	
	*p_points = points;
	*p_searchpoints = searchpoints;
}

int main( int argc, char* argv[] ) {
	srand(0);
	struct Point** points;
	struct Point** searchpoints;
	int i;

	read_input(argc, argv, &points, &searchpoints);
	
	_items = points;
	_root = buildFromPoints(points, 0, npoints);
//	PrintTree(_root, 0, 1);
//	printf("Number of leaves is %d. Max depth is %d, Number of total nodes is %d.\n", leaves, max_depth, number_of_nodes);
//	exit(0);

	if(sort_flag) {
        TIME_START(sort);
		buildFromPoints(searchpoints, 0, npoints);
	    TIME_END(sort);
    }

	TIME_START(init_kernel);
	init_kernel<<<1,1>>>();
	TIME_END(init_kernel);
	TIME_ELAPSED_PRINT(init_kernel, stdout);

	//print_tree(_root);

	int correct_cnt = 0;
	int nsearchpoints = npoints; 	

	int rc;
	pthread_t * threads;
	SAFE_MALLOC(threads, sizeof(pthread_t)*nthreads);
	
	targs * args;
	SAFE_MALLOC(args, sizeof(targs)*nthreads);

	// Assign points to threads
	int start = 0;
	int j;
	for(j = 0; j < nthreads; j++) {
		int num = (npoints - start) / (nthreads - j);
		args[j].searchpoints = searchpoints;
		args[j].tid = j;
		args[j].lb = start;
		args[j].ub = start + num;
		start += num;
		//printf("%d %d\n", args[j].lb, args[j].ub);
	}

	TIME_START(search);

	for(i = 0; i < nthreads; i++) {		
		rc = pthread_create(&threads[i], NULL, search_entry, &args[i]);
		if(rc) {
			fprintf(stderr, "Error: could not create thread, rc = %d\n", rc);
			exit(1);
		}
	}
	
	// wait for threads
	for(i = 0; i < nthreads; i++) {
		pthread_join(threads[i], NULL);
	}

	// compute correct count
	for (i = 0; i < npoints; i++) {
		struct Point *target = searchpoints[i];
		if (target->label == target->closest_label) {
			correct_cnt++;
		}
	}
	
	TIME_END(search);
	TIME_ELAPSED_PRINT(search, stdout);

#ifdef TRACK_TRAVERSALS
	unsigned long long sum_nodes_traversed = 0;
	int sum_trunc = 0;
	int na;
	int maximum = 0, all = 0;
	unsigned long long maximum_sum = 0, all_sum = 0;
	for(i = 0; i < nsearchpoints + (nsearchpoints % 32); i+=32) {
		struct Point* p = searchpoints[i];
		sum_nodes_traversed += p->num_nodes_traversed;
		sum_trunc += p->num_trunc;
		na = p->num_nodes_traversed;    

        for(j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
//          p = searchpoints[j];
            p = searchpoints[j];
            if(p->num_nodes_traversed > na)
               na = p->num_nodes_traversed;
            sum_nodes_traversed += p->num_nodes_traversed;
            sum_trunc += p->num_trunc;

        }

        if (warp_flag) {
            p = searchpoints[i];
            maximum = p->num_nodes_traversed;
            all = p->num_nodes_traversed;
            for(j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
                p = searchpoints[j];

                if (p->num_nodes_traversed > maximum)
                    maximum = p->num_nodes_traversed;
                all += p->num_nodes_traversed;
            }
//          printf("nodes warp %d: %d\n", i/32, na);
            printf("%d\n", maximum);
            maximum_sum += maximum;
            all_sum += all;
        }
	}
	printf("@ maximum_sum: %llu\n", maximum_sum);
	printf("@ all_sum: %llu\n", all_sum);

	printf("@ sum_nodes_traversed: %llu\n", sum_nodes_traversed);
	printf("@ avg_nodes_traversed: %f\n", (float)sum_nodes_traversed / nsearchpoints);
	printf("sum_trunc:%d\n", sum_trunc);
#endif

	float correct_rate = (float) correct_cnt / nsearchpoints;
	printf("correct rate: %.4f\n", correct_rate);

	// TODO: free the rest but its not really important
	/*
	for(i = 0; i < npoints; i++) {
		free(points[i]);
		free(searchpoints[i]);
	}
	free(points);
	free(searchpoints);
	*/
	return 0;
}
