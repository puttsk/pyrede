import json

from pprint import pprint
from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser 
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *

REL_OFFSETS = ['BRA', 'SSY', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

class Sass():
    def __init__(self, sass_file):
        self.header = []
        with open(sass_file) as sass_file:
            for line in sass_file:
                if line[0] == '#':
                    self.header.append(line)
                else:
                    break    
            line = sass_file.readline()
            
            if line != "<CONSTANT_MAPPING>\n":
                raise ValueError("Invalid SASS File")
            
            json_str = ""
            for line in sass_file:
                if line != "</CONSTANT_MAPPING>\n":
                    json_str += line
                else:
                    break
            
            self.constants = json.loads(json_str)
            self.sass_raw = "".join(sass_file.readlines()).strip()    
            
def compile(args):
    sass = Sass(args.input_file)
    program = sass_parser.parse(sass.sass_raw, lexer=sass_lexer)
   
    print("Register usage: %d" % len(program.registers))
    pprint(program.ast)
    
    reg_live_map = dict.fromkeys(program.registers)
    reg_scratch_map = dict.fromkeys(program.registers)
    
    # Build CFG
    
    # Find the beginning of basic blocks. A basic block begin at the start
    # of a program, after a label, or a new predicate is found. 
    leader = []
    read_leader = True
    prev_predicate = None    
    for inst in program.ast:
        if isinstance(inst, Instruction) and read_leader:
            # Mark the instruction as the beginning of a new basic block 
            leader.append(inst)
            prev_predicate = inst.predicate
            read_leader = False
            
        elif isinstance(inst, Instruction) and not read_leader:
            if inst.opcode.name in JUMP_OPS:
                # If face jump instruction, next instruction is the 
                # beginning of a new block
                read_leader = True
             
            if inst.predicate and inst.predicate != prev_predicate:
                # If the instruction has predicate and the predicate is not the 
                # same as previous Instruction, this instruction is the beginning of
                # a new block
                leader.append(inst)
                prev_predicate = inst.predicate
            elif not inst.predicate and prev_predicate != None:
                # If the previous instruction has predicate and the current instruction
                # does not have one, this instruction is the beginning of the block 
                leader.append(inst)
                prev_predicate = None
        elif isinstance(inst, Label):
            read_leader = True
    
    # Construct CFG
    label_table = {} 
    cfg = Cfg()
    for lead_inst in leader:
        next_leader = leader.index(lead_inst)+1 
        
        ast_idx = program.ast.index(lead_inst)
        
        if next_leader < len(leader):
            ast_idx_next = program.ast.index(leader[next_leader])
        else:
            ast_idx_next = len(program.ast)

        if isinstance(program.ast[ast_idx_next -1], Label):
            ast_idx_next -= 1
        
        block = BasicBlock(program.ast[ast_idx:ast_idx_next])        

        if ast_idx > 0 and isinstance(program.ast[ast_idx-1], Label):
            label = program.ast[ast_idx-1]
            block.attach_label(label)
            label_table[label] = block 
        
        # Block appears in its original program order
        cfg.add_basic_block(block)        

                
            
def test_lexer(sass):
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print()
                
        
        
    
    