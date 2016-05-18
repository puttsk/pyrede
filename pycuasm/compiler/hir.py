from enum import Enum

class Flags():
    def __init__(self, wait_barrier, read_barrier, write_barrier, yield_hint, stall):
        
        self.wait_barrier = int(wait_barrier) if wait_barrier != '--' else 0
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

class Opcode():
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Condition():
    def __init__(self, predicate, condition=True):
        self.predicate = predicate
        self.condition = condition
        
    def __str__(self):
        return "@%s%s" % ("" if self.condition else "!", self.predicate)

class Instruction():
    def __init__(self, flags, opcode, operands=None, predicate=None):
        self.flags = flags
        self.opcode = opcode
        self.operands = operands
        self.predicate = predicate
        self.addr = 0
    
    def __str__(self):
        return "%4x: %s %s\t%s %s" % ( self.addr,   
                                    self.flags, 
                                    self.predicate if self.predicate else "", 
                                    self.opcode,
                                    self.operands)

    def __repr__(self):
        return self.__str__()

class Label():
    def __init__(self, name):
        self.name = name
        self.addr = 0

    def __str__(self):
        return "%s:%x" % (self.name, self.addr)
    
    def __repr__(self):
        return self.__str__()
        
class Pointer():
    def __init__(self, register):
        self.register = register
        
    def __str__(self):
        return "[%s]" % self.register

    def __repr__(self):
        return self.__str__()        

class Predicate():
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Register():
    def __init__(self, name, is_special = False):
        name = name.split('.')
        self.name = name[0]
        self.attributes = name[1:]
        self.is_special = is_special

    def __str__(self):
        return "%s" % self.name
    
    def __repr__(self):
        return self.__str__()    
    
    def __eq__(self, other):
        return self.name == other.name
    
class Constant():
    def __init__(self, name, is_param = False):
        self.name = name
        self.is_param = is_param

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()