#ifndef __PTR_TAB_H
#define __PTR_TAB_H

#define PTRTAB_FREE ((void*)(-1))

struct ptrtab_entry {
	void *key;
	int value;
};

struct ptrtab {
	struct ptrtab_entry *items;
	unsigned int count;
	unsigned int size;
};

struct ptrtab*  ptrtab_init(unsigned int init_size);
void ptrtab_free(struct ptrtab *tab);

int ptrtab_find(struct ptrtab *tab, void* key, int *out_val);
int ptrtab_insert(struct ptrtab *tab, void *key, int value);
void ptrtab_clear(struct ptrtab *tab);

unsigned int ptrtab_hash(void* key);

#endif
