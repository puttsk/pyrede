/* -*- mode: c -*- */

#define LEFT cur_node2.items.left
#define RIGHT cur_node2.items.right

#define STACK_INIT()	sp = 1;
#define STACK_PUSH(node) sp = sp + 1; stk[WARP_IDX][sp] = node
#define STACK_POP() sp = sp - 1;
#define STACK_NODE stk[WARP_IDX][sp]
