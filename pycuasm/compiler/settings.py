DEBUG = True

reserved = {
   'blockDimX' : 'BLOCK_DIM_X',
   'blockDimY' : 'BLOCK_DIM_Y',
   'blockDimZ' : 'BLOCK_DIM_Z',
   'gridDimX' : 'GRID_DIM_X',
   'gridDimY' : 'GRID_DIM_Y',
   'gridDimZ' : 'GRID_DIM_Z',
}

# List of token names.   This is always required
tokens = [
   'CONSTANT',
   'FLAGS',
   'ID',
   'HEXADECIMAL',
   'INTEGER',
   'FLOAT',
   'PREDICATE',
   'REGISTER',
   'SPECIAL_REGISTER',
   'PARAMETER'
] + list(reserved.values())

