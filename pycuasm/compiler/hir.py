import json
import copy

from enum import Enum
from pycuasm.compiler.grammar import SASS_GRAMMARS

class Program(object):
    def __init__(self, ast):
        self.ast = ast
        self.constants = None
        self.header = None
        
        self.update()
    
    def update(self):
        #Update AST and build register table
        registers = []
        
        addr = 0x0008
        instCount = 0
        for inst in self.ast:
            if isinstance(inst, Instruction):
                # Assign address to each instruction
                inst.addr = addr
                addr += 8 #Each instruction size is 8-byte long
                instCount += 1
                if instCount % 3 == 0: 
                    addr += 8
                
                regs = [x for x in inst.operands + [inst.dest] if isinstance(x, Register) and not x.is_special]
                registers += sorted([x.name for x in regs if x.name not in registers])
                
            elif isinstance(inst, Label):
                inst.addr = addr
            else:
                raise ValueError("Unknown IR Type: %s %s" % (inst.__class__, inst))

        self.registers = registers
    
    def save(self, outfile):
        f = open(outfile, 'w')
        f.writelines(self.header)
        
        f.write("\n<CONSTANT_MAPPING>\n")
        for const in self.constants:
            f.write("\t%s : %s\n" % (const, self.constants[const])) 
        f.write("</CONSTANT_MAPPING>\n\n")
        
        for inst in self.ast:  
            f.write(str(inst) + '\n')
        f.close()
            

class Instruction(object):
    def __init__(self, flags, opcode, operands=None, condition=None):
        self.flags = flags
        self.opcode = opcode
        self.operands = operands
        self.condition = condition
        self.addr = 0        
        self.dest = None
        
        if opcode.reg_store and isinstance(operands[0], Register):
            self.dest = operands[0] 
            self.operands = operands[1:]
        
    def __str__(self):
        return "%s %s\t%s%s%s;" % (self.flags, 
                                    self.condition if self.condition else "", 
                                    self.opcode,
                                    (" " + str(self.dest) + ',') if self.dest else "",
                                    (" " + ", ".join([str(x) for x in self.operands])) if self.operands else ""
                                    )

    def __repr__(self):
        return "%4x: %s" % (self.addr, self.__str__())
        
    @property
    def reg_store(self):
        return self.opcode.reg_store

class Flags(object):
    def __init__(self, wait_barrier, read_barrier, write_barrier, yield_hint, stall):
        
        self.wait_barrier = int(wait_barrier, 16) if wait_barrier != '--' else 0
        """" Wait Dependency Barrier Flags
        Wait on a previously set barrier of either type. You can wait on more than 
        one barrier at at time so instead of working off the barrier number directly,
        a bit mask is used. Here are the mask values (in hex) and corresponding 
        barrier numbers:
            01 : 1
            02 : 2
            04 : 3
            08 : 4
            10 : 5
            20 : 6
        """
        self.read_barrier = int(read_barrier) if read_barrier != '-' else 0
        self.write_barrier = int(write_barrier) if write_barrier != '-' else 0
        self.yield_hint = True if yield_hint == 'Y' else False
        self.stall = int(stall, 16)
    
    def __str__(self):
        return "%s:%s:%s:%s:%x" % (
            ("%02x" % self.wait_barrier) if self.wait_barrier != 0 else '--',
            self.read_barrier if self.read_barrier != 0 else '-',
            self.write_barrier if self.write_barrier != 0 else '-',
            'Y' if self.yield_hint != 0 else '-',
            self.stall
        )
        
    def __repr__(self):
        return self.__str__()

class Opcode(object):
    def __init__(self, opcode):
        name = opcode.split('.')
        self.name = name[0]
        self.extension = name[1:]        
        self.use_carry_bit = 'X' in self.extension
        
        if self.name not in SASS_GRAMMARS:
            raise ValueError("Invalid instruction: " + name)

        self.grammar = SASS_GRAMMARS[self.name]

    def __str__(self):
        return self.full 
    
    def __repr__(self):
        return self.full 
    
    @property
    def full(self):
        return '.'.join([self.name] + self.extension)
    
    @property
    def reg_store(self):
        return self.grammar.reg_store
        
    @property
    def type(self):
        return self.grammar.type
        
    @property
    def integer_inst(self):
        return self.grammar.integer_inst
        
    @property
    def float_inst(self):
        return self.grammar.float_inst
        
    @property
    def is_64(self):
        return self.grammar.is_64
        
class Condition(object):
    def __init__(self, predicate, condition=True):
        self.predicate = predicate
        self.condition = condition
        
    def __str__(self):
        return "@%s%s" % ("" if self.condition else "!", self.predicate)
    
    def __eq__(self, other):
        return self and other and (self.predicate.name == other.predicate.name and self.condition == other.condition)

class Label(object):
    def __init__(self, name):
        self.name = name
        self.addr = 0

    def __str__(self):
        return "%s:" % self.name
    
    def __repr__(self):
        return "[%d]%s" % (self.addr, self.name)
        
class Pointer(object):
    def __init__(self, register, offset=0, is_64bit=False):
        self.register = register
        self.offset = offset
        self.is_64bit = is_64bit
        
    def __str__(self):
        return "[%s%s]" % (self.register, 
                            ("+" + str(self.offset)) if self.offset != 0 else "")

    def __repr__(self):
        return self.__str__()        

class Predicate(object):
    def __init__(self, name, is_inverse=False):
        self.name = name
        self.is_inverse = is_inverse

    def __str__(self):
        return "%s%s" % ('!' if self.is_inverse else '', self.name)
    
    def __repr__(self):
        return self.__str__()

class Identifier(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Register(object):
    def __init__(self, register, is_special = False, is_negative = False, is_absolute = False):
        name = register.split('.')
        
        self.name = copy.deepcopy(name[0])
        self.extension = copy.deepcopy(name[1:])
        self.reuse = True if 'reuse' in self.extension else False
        self.is_special = is_special
        self.is_negative = is_negative
        self.is_absolute = is_absolute
        
        self.carry_bit = 'CC' in self.extension

    @property
    def full(self):
        return "%s%s%s" % (self.name, '.' if self.extension else '', '.'.join(self.extension))

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "%s%s%s%s%s%s" % ('|' if self.is_absolute else '','-' if self.is_negative else '',self.name, '.' if self.extension else '', '.'.join(self.extension), '|' if self.is_absolute else '')
    
    def __repr__(self):
        return self.full   
    
    def __eq__(self, other):
        if not isinstance(other, Register):
            return False
        return self.name == other.name
    
    def rename(self, new_name):
        if isinstance(new_name, Register):
            self.name = new_name.name
        elif isinstance(name, str):
            self.name = new_name
    
class Constant(object):
    def __init__(self, name, is_param = False):
        self.name = name
        self.is_param = is_param

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()