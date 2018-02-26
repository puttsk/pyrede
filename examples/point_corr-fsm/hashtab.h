#ifndef __HASHTAB_H
#define __HASHTAB_H

#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define MAX_KEY_LEN sizeof(void*)
#define MAX_VALUE_LEN sizeof(void*)

#ifdef __CUDACC__
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

typedef size_t (*hash_fn)(char * key, size_t key_size);

typedef enum _htError {
	htSuccess,
	htTableFull,
	htKeyOverflow,
	htValueOverflow,
	htNotFound
} htError;

typedef struct _hashtab_entry {
	char key[MAX_KEY_LEN];
	size_t key_size;
	char value[MAX_VALUE_LEN];
	size_t value_size;
} *hashtab_entry, hashtab_entry_struct;

typedef struct _hashtab {
	hashtab_entry entries;
	size_t table_size;
	size_t num_entries;
	hash_fn hash_function;
} *hashtab, hashtab_struct;

EXTERN_C size_t  hashtab_default_hash_fn(char * key, size_t key_size);
EXTERN_C size_t hashtab_count(hashtab ht);
EXTERN_C size_t hashtab_isfull(hashtab ht);

EXTERN_C inline size_t hashtab_keyequal(char *k1, char *k2, size_t key_size);

EXTERN_C hashtab hashtab_new(size_t size);
EXTERN_C void hashtab_free(hashtab table);
EXTERN_C htError hashtab_insert(hashtab table, void * key, size_t key_size, void * value, size_t value_size);
EXTERN_C htError hashtab_get(hashtab table, void * key, size_t key_size, void * out_value, size_t * out_value_size);

#endif
