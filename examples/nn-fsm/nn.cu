/* -*- mode: c -*- */
#include "nn.h"
#include "nn_gpu.h"
/* -*- mode: c -*- */

#include <float.h>
#include "gpu_tree.h"

__global__ void init_kernel(void) {

}

__global__ void nearest_neighbor_search (gpu_tree gpu_tree, gpu_point *d_training_points, int n_training_points,
																				 gpu_point *d_search_points, int n_search_points) 
{

 	float search_points_coord[DIM];
	int closest;
	float closestDist;

#ifdef TRACK_TRAVERSALS
	int numNodesTraversed;
#endif

	int i, j, pidx;
	
	int cur_node_index, prev_node_index;
	__shared__ int SP[NUM_WARPS_PER_BLOCK];
#define sp SP[WARP_IDX]
	
	__shared__ int stk[NUM_WARPS_PER_BLOCK][64];
	__shared__ extern int tmp_shr[];
	int stk_top;

	bool cond, status;
    bool opt1, opt2;
	int critical;
	unsigned int vote_left;
	unsigned int vote_right;
	unsigned int num_left;
	unsigned int num_right;

	gpu_tree_node_0 cur_node0;
	gpu_tree_node_2 cur_node2;
	gpu_tree_node_3 cur_node3;

	float dist=0.0;
	float boxdist=0.0;
	float sum=0.0;
	float boxsum=0.0;
	float center=0.0;
	int id = 0;

#include "nn_kernel_macros.inc"

	for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < n_search_points; pidx += blockDim.x * gridDim.x)
    {
		for(j = 0; j < DIM; j++) {
			search_points_coord[j] = d_search_points[pidx].coord[j];
		}

		closest = d_search_points[pidx].closest;
		closestDist = d_search_points[pidx].closestDist;
#ifdef TRACK_TRAVERSALS
		numNodesTraversed = 0; //d_search_points[pidx].numNodesTraversed;
#endif

		cur_node_index = 0;
		STACK_INIT ();
		STACK_NODE = 0;
		status = 1;
		critical = 63;
		cond = 1;

		while(sp >= 1) {
			cur_node_index = STACK_NODE;

//			if (pidx == 0) {
//				printf("status = %d, critical = %d, sp = %d, and cur_node_index = %d\n", status, critical, sp, cur_node_index);
//			}

			if (status == 0 && critical >= sp) {
				status = 1;
			}
			STACK_POP();

			if (status == 1) {
#ifdef TRACK_TRAVERSALS
			numNodesTraversed++;
#endif
//                critical = sp - 1;
//                if (pidx == 0) {
//                    printf("cur_node_index = %d\n", cur_node_index);
//                }
				//cur_node1 = gpu_tree.nodes1[cur_node_index];
				// inlined function can_correlate
				dist=0.0;
				boxdist=0.0;
				sum=0.0;
				boxsum=0.0;
				center=0.0;

				for(i = 0; i < DIM; i++) {
					float max = gpu_tree.nodes1[cur_node_index].items[i].max;
					float min = gpu_tree.nodes1[cur_node_index].items[i].min;
				    center = (max + min) / 2;
					boxdist = (max - min) / 2;
					dist = search_points_coord[i] - center;
					sum += dist * dist;
					boxsum += boxdist * boxdist;
				}

				cond = (sqrt(sum) - sqrt(boxsum) < sqrt(closestDist));
                critical = sp;
				if(!__any(cond)) {
					continue;
				}

//                critical = sp - 1;
				if (!cond) {
					status = 0;
//                    critical = sp - 1;
				} else {
					cur_node0 = gpu_tree.nodes0[cur_node_index];
					if(cur_node0.items.axis == DIM) {
						cur_node3 = gpu_tree.nodes3[cur_node_index];
						for(i = 0; i < MAX_POINTS_IN_CELL; i++) {
							if(cur_node3.points[i] >= 0) {
								// update closest...
								float dist = 0.0;
								float t;

								for(j = 0; j < DIM; j++) {
									t = (d_training_points[cur_node3.points[i]].coord[j] - search_points_coord[j]);
									dist += t*t;
								}

								if(dist <= closestDist) {
									closest = cur_node3.points[i];
									closestDist = dist;
								}
							}
						}

					} else {
						cur_node2 = gpu_tree.nodes2[cur_node_index];
						opt1 = search_points_coord[cur_node0.items.axis] < cur_node0.items.splitval;
						opt2 = search_points_coord[cur_node0.items.axis] >= cur_node0.items.splitval;
						vote_left = __ballot(opt1);
						vote_right = __ballot(opt2);
						num_left = __popc(vote_left);
						num_right = __popc(vote_right);
						// majority vote
						if (num_left > num_right) {
							if(RIGHT != NULL_NODE) { STACK_PUSH(RIGHT); }
							if(LEFT != NULL_NODE) { STACK_PUSH(LEFT); }
						} else {
							if(LEFT != NULL_NODE) { STACK_PUSH(LEFT); }
							if(RIGHT != NULL_NODE) { STACK_PUSH(RIGHT); }
						}
					}
				}
			}
		}

		d_search_points[pidx].closest = closest;
		d_search_points[pidx].closestDist = closestDist;
#ifdef TRACK_TRAVERSALS
		d_search_points[pidx].numNodesTraversed = numNodesTraversed;
#endif

	}
}
 
int sort_flag = 0;
int verbose_flag = 0;
int check_flag = 0;
int ratio_flag = 0;
int warp_flag = 0;

Point *training_points;
KDCell *root;
Point *search_points;

int npoints;
int nsearchpoints;
char *input_file;

static inline float distance_axis(Point *a, Point *b, int axis);
static inline float distance(Point *a, Point *b);

TIME_INIT(runtime);
TIME_INIT(construct_tree);
TIME_INIT(gpu_build_tree);
TIME_INIT(init_kernel);
TIME_INIT(gpu_copy_to);
TIME_INIT(gpu_copy_from);
TIME_INIT(kernel);
TIME_INIT(sort);
TIME_INIT(traversal_time);


static int leaves = 0;
static int max_depth = 0;
static int number_of_nodes = 0;
void PrintTree(KDCell * root, int depth, int id){
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
	printf("NODE = %d, Depth = %d, axis = %d, ", id, depth, root->axis);
	if (root->axis == DIM) {
//	if (root->left == NULL && root->right == NULL) {
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
}

int main(int argc, char **argv) {

	int correct_cnt, i, j;
	unsigned long long sum_nodes_traversed;
	float correct_rate;
	
	struct thread_args *args;
	pthread_t *threads;

	read_input(argc, argv);
	printf("configuration: sort_flag=%d verbose_flag=%d check_flag=%d DIM = %d npoints = %d nsearchpoints = %d\n", sort_flag, verbose_flag, check_flag, DIM, npoints, nsearchpoints);

	TIME_START(runtime);
	TIME_START(construct_tree);

	if(sort_flag) {
		TIME_START(sort);
		sort_points(search_points, 0, nsearchpoints - 1, 0);
		TIME_END(sort);
	}

	root = construct_tree(training_points, 0, npoints - 1, 0, 1);
	
//	PrintTree(root, 0, 1);
//	printf("Number of leaves is %d. Max depth is %d, Number of total nodes is %d.\n", leaves, max_depth, number_of_nodes);
//	exit(0);

	TIME_END(construct_tree);
	TIME_START(traversal_time);
	
	TIME_START(gpu_build_tree);
	gpu_tree *h_tree = gpu_transform_tree(root);
	gpu_point *h_training_points = gpu_transform_points(training_points, npoints);
	gpu_point *h_search_points = gpu_transform_points(search_points, nsearchpoints);
	TIME_END(gpu_build_tree);

	TIME_START(init_kernel);
	init_kernel<<<1,1>>>();
	TIME_END(init_kernel);

	TIME_START(gpu_copy_to);
	gpu_tree *d_tree = gpu_copy_to_dev(h_tree);
	gpu_free_tree_host(h_tree);

	gpu_point *d_training_points = gpu_copy_points_to_dev(h_training_points, npoints);
	gpu_free_points_host(h_training_points);
 
	gpu_point *d_search_points = gpu_copy_points_to_dev(h_search_points, nsearchpoints);
	TIME_END(gpu_copy_to);

	dim3 grid(NUM_THREAD_BLOCKS, 1, 1);
	dim3 block(NUM_THREADS_PER_BLOCK, 1, 1);
	TIME_START(kernel);
	nearest_neighbor_search<<<grid, block, 3840>>>(*d_tree, d_training_points, npoints, d_search_points, nsearchpoints);

	cudaError_t err = cudaThreadSynchronize();
	TIME_END(kernel);

	if(err != cudaSuccess) {
		fprintf(stderr,"Kernel failed with error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	TIME_START(gpu_copy_from);

	// copy back into Points from the 
	gpu_copy_points_to_host(d_search_points, h_search_points, search_points, nsearchpoints);
	
	// free device data
	gpu_free_points_host(h_search_points);
	gpu_free_points_dev(d_search_points);
	gpu_free_points_dev(d_training_points);
	gpu_free_tree_dev(d_tree);
 	
	TIME_END(gpu_copy_from);
	TIME_END(traversal_time);
	TIME_END(runtime);

	correct_cnt = 0;
	for(i = 0; i < nsearchpoints; i++) {
			if(search_points[i].closest >= 0) {
				if (training_points[search_points[i].closest].label == search_points[i].label) {
					correct_cnt++;
				}
		}
	}
	
	correct_rate = (float) correct_cnt / nsearchpoints;
	printf("correct rate: %.4f\n", correct_rate);

	#ifdef TRACK_TRAVERSALS
	sum_nodes_traversed = 0;
    int maximum = 0, all = 0;
    unsigned long long maximum_sum = 0, all_sum = 0;
	for (i = 0; i < nsearchpoints + (nsearchpoints % 32); i+=32) {
		int na =search_points[i].numNodesTraversed;
//        printf("nodes warp %d: %d\n", i/32, na);
        sum_nodes_traversed += search_points[i].numNodesTraversed;

        if (warp_flag) {
            maximum = na;
            all = na;
            for(j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
		    	sum_nodes_traversed += search_points[j].numNodesTraversed;
	    		if(search_points[j].numNodesTraversed)
    				na = search_points[j].numNodesTraversed;
		
                    if(search_points[j].numNodesTraversed > maximum)
                        maximum = search_points[j].numNodesTraversed;
                    all += search_points[j].numNodesTraversed;       
            }

            printf("%d\n", maximum);
            maximum_sum += maximum;
            all_sum += all;
        }
    }	

	printf("avg nodes: %f\n", (float)sum_nodes_traversed / nsearchpoints);

//	sum_nodes_traversed = 0;
//	for (int i = 0; i < nsearchpoints; i++)
//    {
//		sum_nodes_traversed += search_points[i].numNodesTraversed;
//	}
    printf("@ sum_nodes_traversed: %ld\n", sum_nodes_traversed);
	printf("@ avg_nodes_traversed: %f\n", (float)sum_nodes_traversed / nsearchpoints);
	#endif 

	// print results
	if(verbose_flag) {
		for(i = 0; i < nsearchpoints; i++) {
			if(search_points[i].closest >= 0) {
				printf("%d: %d (%2.3f)\n", i, training_points[search_points[i].closest].label, search_points[i].closestDist);
			}
		}
	}
	
	TIME_ELAPSED_PRINT(construct_tree, stdout);
	TIME_ELAPSED_PRINT(gpu_build_tree, stdout);
	TIME_ELAPSED_PRINT(init_kernel, stdout);
	TIME_ELAPSED_PRINT(gpu_copy_to, stdout);
	TIME_ELAPSED_PRINT(kernel, stdout);
	TIME_ELAPSED_PRINT(sort, stdout);
	TIME_ELAPSED_PRINT(gpu_copy_from, stdout);
	TIME_ELAPSED_PRINT(traversal_time, stdout);
	TIME_ELAPSED_PRINT(runtime, stdout);

	return 0;
}

void read_input(int argc, char **argv) {
	unsigned long long i, j, k, c;
	//float min = FLT_MAX;
	//float max = FLT_MIN;
	FILE *in;

	if(argc < 3) {
		fprintf(stderr, "usage: nn [-c] [-v] [-s] <input_file> <npoints> [<nsearchpoints>]\n");
		exit(1);
	}

	while((c = getopt(argc, argv, "cvsrw")) != -1) {
		switch(c) {
		case 'c':
			check_flag = 1;
			break;

		case 'v':
			verbose_flag = 1;
			break;

		case 's':
			sort_flag = 1;
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
	
	for(i = optind; i < argc; i++) {
		switch(i - optind) {
		case 0:
			input_file = argv[i];
			break;

		case 1:
				npoints = atoi(argv[i]);
				nsearchpoints = npoints;
				if(npoints <= 0) {
					fprintf(stderr, "Not enough points.\n");
					exit(1);
				}
				break;

		case 2:
			nsearchpoints = atoi(argv[i]);
			if(nsearchpoints <= 0) {
				fprintf(stderr, "Not enough search points.");
				exit(1);
			}
			break;
		}
	}

	training_points = alloc_points(npoints);
	search_points = alloc_points(nsearchpoints);

	if(strcmp(input_file, "random") == 0) {
		for(i = 0; i < npoints; i++) {
			training_points[i].label = i;
			for(j = 0; j < DIM; j++) {
				training_points[i].coord[j] = 1.0 + (float)rand() / RAND_MAX;			
			}
		}

		for(i = 0; i < nsearchpoints; i++) {
			search_points[i].label = npoints + i;
			for(j = 0; j < DIM; j++) {
				search_points[i].coord[j] = 1.0 + (float)rand() / RAND_MAX;			
			}
		}

	} else {
		in = fopen(input_file, "r");
		if(in == NULL) {
			fprintf(stderr, "Could not open %s\n", input_file);
			exit(1);
		}

		for(i = 0; i < npoints; i++) {
			read_point(in, &training_points[i]);
		}

		for(i = 0; i < nsearchpoints; i++) {
			read_point(in, &search_points[i]);
		}

		fclose(in);
	}
}

Point* alloc_points(int n) {
	int i, j;
	Point *points;
	SAFE_MALLOC(points, sizeof(Point) * n);
	for (i = 0; i < n; i++) {
		points[i].closestDist = FLT_MAX;
		points[i].closest = -1;
		#ifdef TRACK_TRAVERSALS
		points[i].numNodesTraversed = 0;
		#endif
	}
	return points;
}

KDCell* alloc_kdcell() {
	int i;
	KDCell *cell;
	SAFE_MALLOC(cell, sizeof(KDCell));
	for (i = 0; i < DIM; i++) {
		cell->min[i] = FLT_MAX;
		cell->max[i] = FLT_MIN;
	}

	for (i = 0; i < MAX_POINTS_IN_CELL; i++) {
		cell->points[i] = -1;
	}

	cell->left = NULL;
	cell->right = NULL;
	return cell;
}

void read_point(FILE *in, Point *p) {
	int j;
	if(fscanf(in, "%d", &p->label) != 1) {
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}
	for(j = 0; j < DIM; j++) {
		if(fscanf(in, "%f", &p->coord[j]) != 1) {
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}
	}
}
