#include <stdio.h>
#include "hashtab.h"

hashtab hashtab_new(size_t size) {
	size_t i;
	hashtab ht = (hashtab)malloc(sizeof(hashtab_struct));
	ht->entries = (hashtab_entry)malloc(sizeof(hashtab_entry_struct) * size);
	for(i = 0; i < size; i++) {
		ht->entries[i].key_size = 0;
		ht->entries[i].value_size = 0;
	}

	ht->table_size = size;
	ht->num_entries = 0;
	ht->hash_function = hashtab_default_hash_fn;

	return ht;
}

void hashtab_free(hashtab table) {
	free(table->entries);
	free(table);
}

htError hashtab_insert(hashtab table, void * key, size_t key_size, void * value, size_t value_size) {

	if(key_size > MAX_KEY_LEN)
		return htKeyOverflow;

	if(value_size > MAX_VALUE_LEN)
		return htValueOverflow;

	if(hashtab_isfull(table))
		return htTableFull;

	int probe, idx = 0;
	size_t hash = table->hash_function(key, key_size);
	
	idx = hash % table->table_size;
	while(table->entries[idx].key_size != 0) { 
		hash = hash + 1;
		idx = hash % table->table_size;
	} 

	table->entries[idx].key_size = key_size;
	table->entries[idx].value_size = value_size;
	memcpy(table->entries[idx].key, key, key_size);
	memcpy(table->entries[idx].value, value, value_size);

	table->num_entries++;
 
	return htSuccess;
}

htError hashtab_get(hashtab table, void * key, size_t key_size, void * out_value, size_t * out_value_size) {
	
	int probe, idx;
	size_t hash = table->hash_function(key, key_size);
	
	idx = hash % table->table_size;
	while(1) {

		if(table->entries[idx].key_size == 0)
			return htNotFound;

		if(hashtab_keyequal((char*)key, table->entries[idx].key, key_size))
		{
			memcpy(out_value, table->entries[idx].value, table->entries[idx].value_size);
			
			// null if size if fixed or known by user
			if(out_value_size != NULL)
				*out_value_size = table->entries[idx].value_size;

			return htSuccess;
		} 

		probe++;
		hash = hash + 1;
		idx = hash % table->table_size;
		if(probe >= table->table_size)
			return htNotFound;
	}
		
}

inline size_t hashtab_keyequal(char *k1, char *k2, size_t key_size) {
	size_t i;
	for(i = 0; i < key_size; i++) {
		if(k1[i] != k2[i])
			return 0;
	}

	return 1;
}

size_t hashtab_count(hashtab ht) {
	return ht->num_entries;
}

size_t hashtab_isfull(hashtab ht) {
	if(ht->num_entries == ht->table_size)
		return 1;
	else
		return 0;
}


size_t  hashtab_default_hash_fn(char * key, size_t key_size) {
	size_t hash = 0;
	size_t i;
	int c;

	for(i = 0; i < key_size; i++) {
		c = key[i];
		hash = c + (hash << 6) + (hash << 16) - hash;
	}

	return hash;
}
