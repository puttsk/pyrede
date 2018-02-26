/* -*- mode: c++ -*- */
#include <stdio.h>
#include <stdlib.h>
#include "vptree.h"
#include "ptrtab.h"

extern struct Point *__GPU_point_Point_d;
extern struct Point *__GPU_point_Point_h;
extern struct __GPU_point *__GPU_point_array_d;
extern struct __GPU_point *__GPU_point_array_h;
extern struct Point *__GPU_node_point_d;
extern struct Point *__GPU_node_point_h;

void __GPU_findTreeDepth(struct ptrtab *ptab,struct Node *cpu_root,int *nnodes,int *depth,int cur_depth)
{
  if (ptrtab_find(ptab,cpu_root,0)) 
    return ;
  else 
    ptrtab_insert(ptab,cpu_root,0);
  cur_depth++;
  ( *nnodes)++;
  if (cur_depth >  *depth) 
     *depth = cur_depth;
  if (cpu_root -> parent != 0) 
    __GPU_findTreeDepth(ptab,cpu_root -> parent,nnodes,depth,cur_depth);
  if (cpu_root -> left != 0) 
    __GPU_findTreeDepth(ptab,cpu_root -> left,nnodes,depth,cur_depth);
  if (cpu_root -> right != 0) 
    __GPU_findTreeDepth(ptab,cpu_root -> right,nnodes,depth,cur_depth);
}

int __GPU_copyTree(struct ptrtab *ptab,struct Node *cpu_root,struct __GPU_Node *gpu_root,int *index)
{
  int myidx;
  int ch;
  struct __GPU_Node *g;
  if (ptrtab_find(ptab,cpu_root,&ch)) 
    return ch;
  myidx =  *index;
  ptrtab_insert(ptab,cpu_root,myidx);
  ( *index)++;
  g = gpu_root + myidx;
  memcpy(__GPU_node_point_h + myidx,cpu_root -> point,sizeof(struct Point ));
  g -> point = myidx;
  g -> threshold = cpu_root -> threshold;
  if (cpu_root -> parent != 0) {
    ch = __GPU_copyTree(ptab,cpu_root -> parent,gpu_root,index);
    g -> parent = ch;
  }
  else 
    g -> parent = -1;
  if (cpu_root -> left != 0) {
    ch = __GPU_copyTree(ptab,cpu_root -> left,gpu_root,index);
    g -> left = ch;
  }
  else 
    g -> left = -1;
  if (cpu_root -> right != 0) {
    ch = __GPU_copyTree(ptab,cpu_root -> right,gpu_root,index);
    g -> right = ch;
  }
  else 
    g -> right = -1;
  return myidx;
}

struct __GPU_tree __GPU_buildTree(struct Node *cpu_root,int npoints)
{
  struct __GPU_tree gpu_tree;
  struct ptrtab *ptab;
  int index = 0;
  ptab = ptrtab_init(10000);
  gpu_tree.nnodes = 0;
  gpu_tree.depth = 0;
  gpu_tree.npoints = npoints;
  __GPU_findTreeDepth(ptab,cpu_root,&gpu_tree.nnodes,&gpu_tree.depth,0);
  gpu_tree.nodes = ((struct __GPU_Node *)(malloc(sizeof(struct __GPU_Node ) * gpu_tree.nnodes)));
  ptrtab_clear(ptab);
  __GPU_copyTree(ptab,cpu_root,gpu_tree.nodes,&index);
  ptrtab_free(ptab);
  return gpu_tree;
}

struct __GPU_tree __GPU_allocDeviceTree(struct __GPU_tree gpu_tree_h)
{
  struct __GPU_tree gpu_tree_d;
  gpu_tree_d.nnodes = gpu_tree_h.nnodes;
  gpu_tree_d.npoints = gpu_tree_h.npoints;
  gpu_tree_d.depth = gpu_tree_h.depth;
  if (cudaMalloc(((void **)(&gpu_tree_d.nodes)),sizeof(struct __GPU_Node ) * gpu_tree_d.nnodes) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMalloc failed: s%","transformation",0,"cudaMalloc(((void **)(&gpu_tree_d.nodes)),sizeof(struct __GPU_Node ) * gpu_tree_d.nnodes)");
    abort();
  }
  if (cudaMalloc(((void **)(&gpu_tree_d.stack)),sizeof(struct __GPU_stack_item ) * gpu_tree_d.nnodes) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMalloc failed: s%","transformation",0,"cudaMalloc(((void **)(&gpu_tree_d.stack)),sizeof(struct __GPU_stack_item ) * gpu_tree_d.nnodes)");
    abort();
  }
  if (cudaMalloc(((void **)(&__GPU_point_array_d)),sizeof(struct __GPU_point ) * gpu_tree_d.npoints) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMalloc failed: s%","transformation",0,"cudaMalloc(((void **)(&__GPU_point_array_d)),sizeof(struct __GPU_point ) * gpu_tree_d.npoints)");
    abort();
  }
  if (cudaMalloc(((void **)(&__GPU_point_Point_d)),sizeof(struct Point ) * gpu_tree_d.npoints) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMalloc failed: s%","transformation",0,"cudaMalloc(((void **)(&__GPU_point_Point_d)),sizeof(struct Point ) * gpu_tree_d.npoints)");
    abort();
  }
  if (cudaMalloc(((void **)(&__GPU_node_point_d)),sizeof(struct Point ) * gpu_tree_d.npoints) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMalloc failed: s%","transformation",0,"cudaMalloc(((void **)(&__GPU_node_point_d)),sizeof(struct Point ) * gpu_tree_d.npoints)");
    abort();
  }
  return gpu_tree_d;
}

void __GPU_memcpyTreeToDev(struct __GPU_tree gpu_tree_h,struct __GPU_tree gpu_tree_d)
{
  if (cudaMemcpy(((void *)gpu_tree_d.nodes),((void *)gpu_tree_h.nodes),sizeof(struct __GPU_Node ) * gpu_tree_d.nnodes,cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMemcpy failed: s%","transformation",0,"cudaMemcpy(((void *)gpu_tree_d.nodes),((void *)gpu_tree_h.nodes),sizeof(struct __GPU_Node ) * gpu_tree_d.nnodes,cudaMemcpyHostToDevice)");
    abort();
  }
  if (cudaMemcpy(((void *)__GPU_point_array_d),((void *)__GPU_point_array_h),sizeof(struct __GPU_point ) * gpu_tree_d.npoints,cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMemcpy failed: s%","transformation",0,"cudaMemcpy(((void *)__GPU_point_array_d),((void *)__GPU_point_array_h),sizeof(struct __GPU_point ) * gpu_tree_d.npoints,cudaMemcpyHostToDevice)");
    abort();
  }
  if (cudaMemcpy(((void *)__GPU_point_Point_d),((void *)__GPU_point_Point_h),sizeof(struct Point ) * gpu_tree_d.npoints,cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMemcpy failed: s%","transformation",0,"cudaMemcpy(((void *)__GPU_point_Point_d),((void *)__GPU_point_Point_h),sizeof(struct Point ) * gpu_tree_d.npoints,cudaMemcpyHostToDevice)");
    abort();
  }
  if (cudaMemcpy(((void *)__GPU_node_point_d),((void *)__GPU_node_point_h),sizeof(struct Point ) * gpu_tree_d.npoints,cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMemcpy failed: s%","transformation",0,"cudaMemcpy(((void *)__GPU_node_point_d),((void *)__GPU_node_point_h),sizeof(struct Point ) * gpu_tree_d.npoints,cudaMemcpyHostToDevice)");
    abort();
  }
}

void __GPU_memcpyTreeToHost(struct __GPU_tree gpu_tree_h,struct __GPU_tree gpu_tree_d) {

  if (cudaMemcpy(((void *)__GPU_point_Point_h),((void *)__GPU_point_Point_d),sizeof(struct Point ) * gpu_tree_d.npoints,cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaMemcpy failed: s%","transformation",0,"cudaMemcpy(((void *)__GPU_point_Point_d),((void *)__GPU_point_Point_h),sizeof(struct Point ) * gpu_tree_d.npoints,cudaMemcpyHostToDevice)");
    abort();
  }
}

void __GPU_freeDeviceTree(struct __GPU_tree gpu_tree_d)
{
  if (cudaFree(((void *)gpu_tree_d.nodes)) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaFree failed: s%","transformation",0,"cudaFree(((void *)gpu_tree_d.nodes))");
    abort();
  }
  if (cudaFree(((void *)gpu_tree_d.stack)) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaFree failed: s%","transformation",0,"cudaFree(((void *)gpu_tree_d.stack))");
    abort();
  }
  if (cudaFree(((void *)__GPU_point_array_d)) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaFree failed: s%","transformation",0,"cudaFree(((void *)__GPU_point_array_d))");
    abort();
  }
  if (cudaFree(((void *)__GPU_point_Point_d)) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaFree failed: s%","transformation",0,"cudaFree(((void *)__GPU_point_Point_d))");
    abort();
  }
  if (cudaFree(((void *)__GPU_node_point_d)) != cudaSuccess) {
    fprintf(stderr,"error [file=%s line=%d]: cudaFree failed: s%","transformation",0,"cudaFree(((void *)__GPU_node_point_d))");
    abort();
  }
}
