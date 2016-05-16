import ply.yacc as yacc

from pycuasm.compiler.settings import tokens
from pycuasm.compiler.hir import *

def p_flags(p):
    '''flags : FLAGS'''
    
    f = p[1].split(':')
    p[0] = Flags(f[0], f[1], f[2], f[3], f[4])
    print(p[0])

def p_operand(p):
    '''operand : REGISTER
               | CONSTANT
               | SPECIAL_REGISTER
               | PARAMETER
               | ID'''
    p[0] = p[1]

# Build the parser
sass_parser = yacc.yacc()