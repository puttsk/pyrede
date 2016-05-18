import ply.yacc as yacc

from pprint import pprint

from pycuasm.compiler.settings import tokens, DEBUG
from pycuasm.compiler.hir import *

def p_program(p):
    '''program  : instruction_list
    '''
    p[0] = p[1]
    
def p_instruction_list(p):
    '''instruction_list : instruction
                        | instruction_list instruction 
    '''
    if len(p) == 2 :
        p[0] = [p[1]]
    else:
        p[1].append(p[2])
        p[0] = p[1]
    
def p_instruction(p):
    '''instruction      : unconditional_instruction
                        | conditional_instruction
                        | label
    '''
    p[0] = p[1]

def p_unconditional_instruction(p):
    '''unconditional_instruction : flags ID operand_list ';'
    '''
    p[0] = Instruction(p[1], Opcode(p[2]),operands=p[3])

def p_conditional_instruction(p):
    '''conditional_instruction  : flags condition ID operand_list ';' 
    '''
    p[0] = Instruction(p[1], Opcode(p[3]),operands=p[4], predicate=p[2])

def p_label(p):
    '''label : ID ':'
    '''
    p[0] = Label(p[1])

def p_flags(p):
    '''flags : FLAGS
    '''
    f = p[1].split(':')
    p[0] = Flags(f[0], f[1], f[2], f[3], f[4])
       
def p_condition(p):
    '''condition    : '@' PREDICATE
                    | '@' '!' PREDICATE
    '''
    if len(p) == 3:
        p[0] = Condition(Predicate(p[2]))
    else:
        p[0] = Condition(Predicate(p[3]), condition=False)
        
def p_operand_list(p):
    '''operand_list : operand
                    | operand_list ',' operand 
                    |
    '''
    if len(p) == 2 :
        p[0] = [p[1]]
    elif len(p) == 1:
        p[0] = []
    else:
        p[1].append(p[3])
        p[0] = p[1]

def p_operand(p):
    '''operand : register
               | special_register
               | immediate
               | pointer
               | ID
               | predicate
               | parameter
               | constant
               | '{' INTEGER ',' INTEGER '}'
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[1:len(p)]

def p_predicate(p):
    '''predicate : PREDICATE
    '''
    p[0] = Predicate(p[1])

def p_constant(p):
    '''constant : CONSTANT
    '''
    p[0] = Constant(p[1])
    
def p_parameter(p):
    '''parameter : PARAMETER
    '''
    p[0] = Constant(p[1], is_param = True)

def p_special_register(p):
    '''special_register : SPECIAL_REGISTER
                        | GRID_DIM_X
                        | GRID_DIM_Y
                        | GRID_DIM_Z
                        | BLOCK_DIM_X
                        | BLOCK_DIM_Y
                        | BLOCK_DIM_Z                        
    '''
    p[0] = Register(p[1], is_special = True)

def p_register(p):
    '''register : REGISTER
    '''
    p[0] = Register(p[1])

def p_pointer(p):
    '''pointer      : '[' register ']'
    '''
    p[0] = Pointer(p[2])

def p_immediate(p):
    '''immediate    : immediate_int
                    | immediate_float
                    | immediate_hex
    '''
    p[0] = p[1]

def p_immediate_int(p):
    '''immediate_int : INTEGER
    '''
    p[0] = int(p[1])
 
def p_immediate_float(p):
    '''immediate_float : FLOAT
    '''
    p[0] = float(p[1])

def p_immediate_hex(p):
    '''immediate_hex : HEXADECIMAL
    '''
    p[0] = int(p[1], 16) 

def p_error(p):
    print("Syntax error at line " + str(p.lineno))
    raise SyntaxError(p)

# Build the parser
sass_parser = yacc.yacc(debug=DEBUG)