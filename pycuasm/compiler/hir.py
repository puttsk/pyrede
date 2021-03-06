import json
import copy
import re

from pprint import pprint

from enum import Enum
from pycuasm.compiler.grammar import SASS_GRAMMARS

class Program(object):
    def __init__(self, ast):
        self.ast = ast
        self.constants = None
        self.header = None
        self.shared_size = 0
        self.update()
        
    def set_constants(self, constants):
        self.constants = constants
    
    def set_header(self, header):
        self.header = header
        for param in self.header:
            if "SharedSize" in param:
                match = re.search("(?P<size>\d+)", param)
                self.shared_size = int(match.group('size'))
        #print("shared_size:" +  str(self.shared_size));
    
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
                ptrs = [x.register for x in inst.operands + [inst.dest] if isinstance(x, Pointer)]
                
                if inst.opcode.name == 'LDG' and inst.opcode.op_bit > 32:
                    reg_count = int(inst.opcode.op_bit / 32)
                    reg = inst.dest
                    reg_id = int(reg.name.replace('R',''))
                    
                    for i in range(1, reg_count):
                        regs.append(Register('R%d' % (reg_id + i))) 
                
                registers += sorted([x.name for x in regs if x.name not in registers])
                registers += sorted([x.name for x in ptrs if x.name not in registers])
                
            elif isinstance(inst, Label):
                inst.addr = addr
            else:
                raise ValueError("Unknown IR Type: %s %s" % (inst.__class__, inst))

        self.registers = sorted(registers, key= lambda x: int(x.replace('R','')))
    
    def save(self, outfile):
        f = open(outfile, 'w')
        f.writelines(self.header)
        
        f.write("\n<CONSTANT_MAPPING>\n")
        for const in sorted(self.constants.keys()):
            f.write("\t%s : %s\n" % (const, self.constants[const])) 
        f.write("</CONSTANT_MAPPING>\n\n")
        
        for inst in self.ast:  
            f.write(str(inst) + '\n')
        f.close()

        print("[CUASM] %s Saved." % outfile)
            

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
        if self.opcode.name != "DEPBAR":
            return "%s %s\t%s%s%s;" % (self.flags, 
                                       self.condition if self.condition else "", 
                                       self.opcode,
                                       (" " + str(self.dest) + ',') if self.dest else "",
                                       (" " + ", ".join([str(x) for x in self.operands])) if self.operands else ""
                                       )
        else:
            # For example DEPBAR.LE SB5, 0x4;
            if self.opcode.extension:
                return "%s\t%s %s;" % (self.flags, self.opcode, 
                                     (", ".join([str(x) for x in self.operands])))
            else:    
                return "%s\t%s {%s};" % (self.flags, self.opcode, 
                                     (",".join([str(x) for x in self.operands[0][1] if x != '{' and x != '}'])))

    def __repr__(self):
        return "%4x: %s" % (self.addr, self.__str__())
        
    @property
    def reg_store(self):
        return self.opcode.reg_store
        
    def has_register(self, reg):
        if isinstance(reg, str):
            reg = Register(reg)
        
        if self.opcode.type == 'x64':
            reg_id = int(int(reg.name.replace('R','')) / 2) * 2
            reg = Register('R%d' % reg_id)
        
        if self.dest and self.dest == reg:
            return True
        
        for op in self.operands:
            if isinstance(op, Pointer):
                if op.register == reg:
                    return op.register == reg
            elif isinstance(op, Constant) and op.pointer:
                if op.pointer.register == reg:
                    return op.register == reg
            elif op == reg:
                return True
        return False

class SpillInstruction(Instruction):
    pass

class SpillLoadInstruction(SpillInstruction):
    def __init__(self, flags, opcode, operands=None, condition=None):
        super(SpillInstruction, self).__init__(flags, opcode, operands, condition)
        self.shared_pointer = self.operands[0]
        self.shared_offset = self.operands[0].offset
        
    @property
    def spill_reg(self):
        return self.dest
    
class SpillStoreInstruction(SpillInstruction):
    def __init__(self, flags, opcode, operands=None, condition=None):
        super(SpillInstruction, self).__init__(flags, opcode, operands, condition)
        self.spill_reg = self.operands[1]
        self.shared_pointer = self.operands[0]
        self.shared_offset = self.operands[0].offset

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
    
    @property
    def wait_barrier_list(self):
        b_list = []
        for barrier in range(6):
            if self.wait_barrier & (1 << barrier):
                b_list.append(barrier + 1)
        return b_list
    
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
        self.op_bit = 32
        
        for ext in self.extension:
            if ext.isdigit():
                self.op_bit = int(ext)
        
        if self.name not in SASS_GRAMMARS:
            raise ValueError("Invalid instruction: " + opcode)
        
        # Setting operation bit
        if self.name == 'XMAD':
            self.op_bit = 16
        
        if self.grammar.is_64:
            self.op_bit = 64 
        
        for extension in self.extension:
            if '64' in extension:            
                self.op_bit = 64
            elif '128' in extension:
                self.op_bit = 128
            elif '16' in extension:
                self.op_bit = 16     

            # TODO: This is a quick fix for shift.u64.hi
            if ('U64' == extension and 'HI' in self.extension or 
                self.name == 'SHF'
            ):
                self.op_bit = 32
                #pprint(self)
        # TODO: This is a quick fix for shift.u64.hi
        if self.name == 'MUFU':
            self.op_bit = 32

    def __str__(self):
        return self.full 
    
    def __repr__(self):
        return self.full 
    
    @property
    def grammar(self):
        return SASS_GRAMMARS[self.name]
    
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
        return self.grammar.is_64 or (self.op_bit == 64)
        
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
        
class StaticPointer(object):
    def __init__(self, offset=0, is_64bit=False):
        self.offset = offset
        self.is_64bit = is_64bit
        
    def __str__(self):
        return "[%s]" % (self.offset)

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
    def __init__(self, name, sign = None):
        self.name = name
        self.sign = sign

    def __str__(self):
        return "%s%s%s" % (self.sign if self.sign else '', self.name, ' ' if self.name == 'INF' else '')
    
    def __repr__(self):
        return self.__str__()

class Register(object):
    def __init__(self, register, is_special = False, is_negative = False, is_absolute = False, sign=None):
        name = register.split('.')
        
        self.name = copy.deepcopy(name[0])
        self.extension = copy.deepcopy(name[1:])
        self.reuse = True if 'reuse' in self.extension else False
        self.is_special = is_special
        self.is_negative = is_negative
        self.is_absolute = is_absolute
        self.sign = sign
        
        self.carry_bit = 'CC' in self.extension
        self.offset = 0

    @property
    def register(self):
        return self.name

    @property
    def full(self):
        return "%s%s%s" % (self.name, '.' if self.extension else '', '.'.join(self.extension))

    @property
    def id(self):
        if not self.is_special:
            return int(self.name.replace('R',''))
        else:
            return -1

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "%s%s%s%s%s%s%s%s" % (
                                   '-' if self.is_negative else '',
                                   '|' if self.is_absolute else '',
                                   self.sign if self.sign else '',
                                   self.name,
                                   '|' if self.is_absolute else '', 
                                   '.' if self.extension else '', 
                                   '.'.join(self.extension),
                                   '' if self.offset == 0 else ' ' + str(self.offset) )
    
    def __repr__(self):
        return self.full   
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if not isinstance(other, Register):
            return False
        return self.name == other.name
    
    def rename(self, new_name):
        if isinstance(new_name, Register):
            self.name = new_name.name
        elif isinstance(name, str):
            self.name = new_name
    
    def add_offset(self, offset):
        self.offset = offset
    
class Constant(object):
    def __init__(self, name, is_param = False, is_negative = False, sign=None, extension = '', pointer=None):
        self.name = name
        self.is_param = is_param
        self.is_negative = is_negative
        self.sign = sign
        self.extension = extension
        self.pointer = pointer

    def __str__(self):
        return "%s%s%s%s%s" % (   '-' if self.is_negative else '', 
                                self.sign if self.sign else '',
                                self.name,
                                self.pointer if self.pointer else '', 
                                self.extension)
    
    def __repr__(self):
        return self.__str__()