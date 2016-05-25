from enum import Enum
from pycuasm.compiler.grammar import SASS_GRAMMARS

class Program(object):
    def __init__(self, ast):
        self.ast = ast
        
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
                registers += [x for x in regs if x not in registers]
                
            elif isinstance(inst, Label):
                inst.addr = addr
            else:
                raise ValueError("Unknown IR Type: %s %s" % (inst.__class__, inst))

        self.registers = registers

class Instruction(object):
    def __init__(self, flags, opcode, operands=None, condition=None):
        self.flags = flags
        self.opcode = opcode
        self.operands = operands
        self.condition = condition
        self.addr = 0        
        self.dest = None
        
        if opcode.reg_store:
            self.dest = operands[0]
            self.operands = operands[1:]
        
    def __str__(self):
        return "%4d: %s %s\t%s %s %s" % ( self.addr,   
                                    self.flags, 
                                    self.condition if self.condition else "", 
                                    self.opcode,
                                    self.dest if self.dest else "",
                                    self.operands)

    def __repr__(self):
        return self.__str__()

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
        return "%d:%d:%d:%d:%x" % (
            self.wait_barrier,
            self.read_barrier,
            self.write_barrier,
            self.yield_hint,
            self.stall
        )

class Opcode(object):
    def __init__(self, opcode):
        name = opcode.split('.')
        self.name = name[0]
        self.extension = name[1:]        
        
        if self.name not in SASS_GRAMMARS:
            raise ValueError("Invalid instruction: " + name)

        self.grammar = SASS_GRAMMARS[self.name]

    def __str__(self):
        return self.full 
    
    def __repr__(self):
        return self.full 
    
    @property
    def full(self):
        return self.name + '.'.join(self.extension)
    
    @property
    def reg_store(self):
        return self.grammar.reg_store

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
        return "[%x]%s" % (self.addr, self.name)
    
    def __repr__(self):
        return self.__str__()
        
class Pointer(object):
    def __init__(self, register):
        self.register = register
        
    def __str__(self):
        return "[%s]" % self.register

    def __repr__(self):
        return self.__str__()        

class Predicate(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Register(object):
    def __init__(self, register, is_special = False, is_negative = False):
        name = register.split('.')
        
        self.full = register
        self.name = name[0]
        self.extension = name[1:]
        self.reuse = True if 'reuse' in self.extension else False
        self.is_special = is_special
        self.is_negative = is_negative

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "%s" % self.name
    
    def __repr__(self):
        return self.full   
    
    def __eq__(self, other):
        return self.name == other.name
    
class Constant(object):
    def __init__(self, name, is_param = False):
        self.name = name
        self.is_param = is_param

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()