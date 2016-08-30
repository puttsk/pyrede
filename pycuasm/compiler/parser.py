import ply.yacc as yacc

from pprint import pprint

from pycuasm.compiler.settings import tokens, DEBUG
from pycuasm.compiler.hir import *

def p_program(p):
    '''program  : instruction_list
    '''
    p[0] = Program(p[1])
    
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
    p[0] = Instruction(p[1], Opcode(p[3]),operands=p[4], condition=p[2])

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
               | identifier
               | predicate
               | parameter
               | constant
               | '{' integer_list '}'
               | register immediate
    '''
    # TODO: Currently, instruction like BRX R4-0x0c is match to 'register immediate', which isn't correct
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 3:
        p[1].add_offset(p[2])
        p[0] = p[1] 
    else:
        p[0] = p[1:len(p)]

def p_integter_list(p):
    '''integer_list : INTEGER
                    | integer_list ',' INTEGER 
    '''
    if len(p) == 2 :
        p[0] = [p[1]]
    else:
        p[1].append(p[3])
        p[0] = p[1]
    
def p_identifier(p):
    '''identifier : ID
                  | '+' ID
                  | '-' ID
    '''
    if len(p) == 2:
        p[0] = Identifier(p[1])
    else:
        p[0] = Identifier(p[2], sign=p[1])
        
def p_predicate(p):
    '''predicate : PREDICATE
                 | '!' PREDICATE
    '''
    if len(p) == 2:
        p[0] = Predicate(p[1])
    else:
        p[0] = Predicate(p[2], is_inverse=True)

def p_constant(p):
    '''constant : CONSTANT
                | '-' CONSTANT
                | ID '[' immediate_hex ']' pointer
    '''
    # TODO: SASS allows indirect access to constant memory e.g. c[0x00][R4]
    #       Need to have a better parser for this.
    if len(p) == 2:
        p[0] = Constant(p[1])
    elif len(p) == 3:
        p[0] = Constant(p[2], is_negative=True)
    else:
        p[0] = Constant(p[1] + p[2] + str(p[3]) + p[4] + str(p[5]))
    
def p_parameter(p):
    '''parameter : PARAMETER
                 | '-' PARAMETER
                 | '~' PARAMETER
    '''
    if len(p) == 2:
        p[0] = Constant(p[1], is_param = True)
    else:
        if p[1] == '-':
            p[0] = Constant(p[2], is_param = True, is_negative=True)
        else:
            p[0] = Constant(p[2], is_param = True, sign=p[1])

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
                | '|' REGISTER '|'
                | '+' REGISTER
                | '-' REGISTER
                | '~' REGISTER  
                | '|' REGISTER '|' EXTENSION
    '''
    if len(p) == 2:
        p[0] = Register(p[1])
    else:
        if p[1] == '-':
            p[0] = Register(p[2], is_negative = True)
        elif p[1] == '|' and p[3] == '|':
            if len(p) == 5:
                p[0] = Register(p[2]+p[4], is_absolute = True)
            else: 
                p[0] = Register(p[2], is_absolute = True)
        elif p[1] == '+':
            p[0] = Register(p[2])
        elif p[1] == '~':
            p[0] = Register(p[2], sign = p[1])
        else:
            raise SyntaxError(p)

def p_pointer(p):
    '''pointer      : '[' register ']'
                    | '[' register '+' immediate ']'
                    | immediate_pointer
    '''
    if len(p) == 4:
        p[0] = Pointer(p[2])
    elif len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = Pointer(p[2], p[4])

def p_immediate_pointer(p):
    '''immediate_pointer    : '[' immediate ']'
                            | '[' ID ']'
    '''
    p[0] = StaticPointer(p[2])

def p_immediate(p):
    '''immediate    : immediate_int
                    | immediate_float
                    | immediate_hex
    '''
    p[0] = p[1]

def p_immediate_int(p):
    '''immediate_int : INTEGER
                     | INTEGER_NEG
    '''
    p[0] = p[1]
 
def p_immediate_float(p):
    '''immediate_float : FLOAT
    '''
    p[0] = p[1]

def p_immediate_hex(p):
    '''immediate_hex : HEXADECIMAL
    '''
    p[0] = p[1]

def p_error(p):
    print("Syntax error at line " + str(p.lineno))
    raise SyntaxError(p)

# Build the parser
sass_parser = yacc.yacc(debug=DEBUG)