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
   
    print("Register usage: %s" % program.registers)
    pprint(program.ast)
    
    reg_live_map = dict.fromkeys(program.registers)
    reg_scratch_map = dict.fromkeys(program.registers)
    
    # Build CFG
    print("\nCrateing CFG")
    
    # Find the beginning of basic blocks. A basic block begin at the start
    # of a program, after a label, or a new predicate is found. 
    leader = []
    read_leader = True
    for inst in program.ast:
        if isinstance(inst, Instruction) and read_leader:
            # Mark the instruction as the beginning of a new basic block 
            leader.append(inst)
            prev_condition = inst.condition
            read_leader = False
            
        elif isinstance(inst, Instruction) and not read_leader:             
            if inst.opcode.name in JUMP_OPS:
                read_leader = True
                
        elif isinstance(inst, Label):
            read_leader = True
    
    # Construct CFG basic blocks
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
        
        block = BasicBlock(program.ast[ast_idx:ast_idx_next],)        
        cfg.add_basic_block(block)
        
        if ast_idx > 0 and isinstance(program.ast[ast_idx-1], Label):
            label = program.ast[ast_idx-1]
            block.attach_label(label)
            label_table[label.name] = block 
        
        # Block appears in its original program order        
    
    # Connect blocks in CFG
    for block in cfg.blocks:
        idx = cfg.blocks.index(block)
        last_inst = block.instructions[-1]
         
        if last_inst.opcode.name not in JUMP_OPS and idx < len(cfg.blocks)-1:
            block.taken = cfg.blocks[idx+1]
        elif last_inst.opcode.name in JUMP_OPS:
            if block.condition:
                if block.condition.condition:
                    block.taken = label_table[last_inst.operands[0]] 
                    block.not_taken = cfg.blocks[idx+1] if idx < len(cfg.blocks)-1 else None
                else:
                    block.not_taken = label_table[last_inst.operands[0]] 
                    block.taken = cfg.blocks[idx+1] if idx < len(cfg.blocks)-1 else None
            else:
                block.taken = label_table[last_inst.operands[0]]

    for block in cfg.blocks:
        print("%s\n\t%s\n\t%s" % (block, block.taken, block.not_taken))
            
def test_lexer(sass):
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print()
                
        
        
    
    