import ply.lex as lex

from pycuasm.compiler.settings import tokens, reserved

literals = [',', ';','[',']', '{', '}', '@', '!', ':', '+', '-', '|', '~']

flag = r'[0-9,a-f,A-F,-]'

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'
t_ignore_COMMENT = r'\#.*'

def t_CONSTANT(t): 
    r'c\[0x[0-9abcdefABCDEF]+\][ ]?\[0x[0-9abcdefABCDEF]+\]'
    return t

def t_HEXADECIMAL(t):
    r'[-]?0x[0-9abcdefABCDEF]+(\.NEG)?'
    return t

def t_FLAGS(t):
    r'([0-9a-fA-F-][0-9a-fA-F-]):([0-9a-fA-F-]):([0-9a-fA-F-]):([Y-]):([0-9a-fA-F-])'
    return t

def t_PARAMETER(t):
    r'param_\d+(\[\d+\])?'
    return t
    
def t_SPECIAL_REGISTER(t):
    r'SR_\w+(\.\w+)?'
    return t

def t_REGISTER(t):
    r'R\d+(\.\w+)?(\.\w+)?'
    return t
    
def t_EXTENSION(t):
    r'\.\w+'
    return t

def t_PREDICATE(t):
    r'P[\d+|T]'
    return t

def t_FLOAT(t):
    r'[-]?\d+\.\d+(e[\+-]\d+)?'
    #t.value = float(t.value)    
    return t

def t_TEX2D(t):
    r'2D'
    t.type = 'ID'
    return t

def t_INTEGER_NEG(t):
    r'[-]?\d+(\.NEG)'
    #t.value = -int(t.value.replace('.NEG',''))    
    return t
    
def t_INTEGER(t):
    r'[-]?\d+'
    #t.value = int(t.value)    
    return t

def t_ID(t):
    r'\w+(\.\w+)*(\[\d+\])?'
    t.type = reserved.get(t.value,'ID')    # Check for reserved words
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    
# Error handling rule
def t_error(t):
    print("Illegal character '%s' at line %d" % (t.value[0], t.lexer.lineno))
    exit(1)

sass_lexer = lex.lex()