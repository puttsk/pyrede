#include "ptrtab.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

struct ptrtab*  ptrtab_init(unsigned int init_size) {
	unsigned int i;

	struct ptrtab *tab = (struct ptrtab*)malloc(sizeof(struct ptrtab));
	if(!tab) {
		fprintf(stderr, "error: could not allocate ptrtab!\n");
		abort();
	}

	tab->size = init_size;
	tab->items = (struct ptrtab_entry*)malloc(sizeof(struct ptrtab_entry) * init_size);
	if(!tab->items) {
		fprintf(stderr, "error: could not allocate ptrtab items!\n");
		abort();
	}

	for(i = 0; i < init_size; i++) {
		tab->items[i].key = PTRTAB_FREE;
	}

	tab->count = 0;

	return tab;
}

void ptrtab_free(struct ptrtab *tab) {
	assert(tab != NULL);
	free(tab->items);
	free(tab);
}

int ptrtab_find(struct ptrtab *tab, void* key, int *out_val) {
	
	assert(tab);
	
	unsigned int i;
	unsigned int k = ptrtab_hash(key) % tab->size;

	for(i = 0; i < tab->size;) {
		if(tab->items[k].key == key) {
			if(out_val != NULL)
				*out_val = tab->items[k].value;
			return 1; // found 
		} else if(tab->items[k].key == PTRTAB_FREE) {
			return 0; // not found
		}

		i++;
		k = k + (i*i);
		k = k % tab->size;
	}

	return 0;
}

int ptrtab_insert(struct ptrtab *tab, void *key, int value) {
	
	assert(tab);
 
	unsigned int i;
	unsigned int k = ptrtab_hash(key) % tab->size;
	for(i = 0; i < tab->size;) {
		if(tab->items[k].key == PTRTAB_FREE) {
			tab->items[k].key = key;
			tab->items[k].value = value;
			return 1; // found 
		}

		i++;
		k = k + (i*i);
		k = k % tab->size;
	}

	// out of space! grow the table!
	// just create a new one and reinsert everything...
	struct ptrtab *new_tab = ptrtab_init(tab->size * 2);
	for(i = 0; i < tab->size; i++) {
		if(tab->items[i].key != PTRTAB_FREE) {
			ptrtab_insert(new_tab, tab->items[i].key, tab->items[i].value);
		}
	}

	free(tab->items); // free old items

	// copy back details
	tab->items = new_tab->items;
	tab->size = new_tab->size;
	
	free(new_tab); // free new table handle 

	// finally insert the new item:
	return ptrtab_insert(tab, key, value);
}

void ptrtab_clear(struct ptrtab *tab) {

	assert(tab);

	unsigned int i;
	for(i = 0; i < tab->size; i++) {
		tab->items[i].key = PTRTAB_FREE;
	}
}

unsigned int ptrtab_hash(void* key) {
#ifdef __LP64__
	unsigned long long keyl = (unsigned long long)key;
	unsigned int hashU = (unsigned int)((unsigned long long)keyl >> 32);
	unsigned int hashL = (unsigned int)(keyl);
	
	hashU = (hashU >> 3) * 2654435761;
	hashL = (hashL >> 3) * 2654435761;
	
	return hashU * 51 + hashL * 17;
#else
	unsigned int hashL = (unsigned int)(key);

	hashL = (hashL >> 3) * 2654435761;
	
	return hashL * 17;
#endif
}

