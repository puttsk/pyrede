import re
import pprint
import json
import sys
import subprocess

from collections import namedtuple, OrderedDict
from pycuasm.cubin import Cubin

CUDA_BIN_DIR = "/usr/local/cuda-6.5/bin/"

REL_OFFSETS = ['BRA', 'SSY', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

class Instruction():
    """
    Instruction class.

    Properties:
        'num': SASS line number
        'pred': Predicate
        'op': Instruction operator
        'ins': Instruction string
        'code': Instruction in raw format
        'flag': Control flags
    """
    def __init__(self, line, pred, opcode, inst, raw, flags="", reuse=""):
        self.__line = line
        self.__pred = pred
        self.__opcode = opcode
        self.__inst = inst
        self.__raw = raw
        self.__flags = flags
        self.__reuse = reuse

    @property
    def line(self):
        return self.__line
    
    @property
    def pred(self):
        return self.__pred
        
    @property
    def opcode(self):
        return self.__opcode
        
    @property
    def inst(self):
        return self.__inst
    @inst.setter
    def inst(self, val):
        self.__inst = val
        
    @property
    def raw(self):
        return self.__raw
        
    @property
    def flags(self):
        return self.__flags
    @flags.setter
    def flags(self, val):
        self.__flags = val
        
    @property
    def reuse(self):
        return self.__reuse
    @reuse.setter
    def reuse(self, val):
        self.__reuse = val


def isSassCtrlLine(line):
    """
    Check if the input line is control instruction.
    
    Args:
        line (str): a SASS line from cuobjdump
    
    Returns:
        bool: True if the line is SASS control instruction, False otherwise. 
        
    """
    m = re.search("^\s+\/\* (?P<CtrlHex>0x[0-9a-f]+)", line)
    if m:
        return True
    else:
        return False

def processSassCtrlFlags(line):
    """
    Extract control and reuse flags from SASS control instruction.
        
    Args:
        line (str): a SASS control line from cuobjdump
    
    Returns:
        {
            'ctrl': A set of control flags
            'reuse': A set of reuse flags
        } 
    
    Raises:
        ValueError: If line is not control SASS line   
    """
    m = re.search("^\s+\/\* (?P<CtrlHex>0x[0-9a-f]+)", line)
    
    Flags = namedtuple('Flags', ['ctrl', 'reuse'])
    
    if m:
        ctrlInst = int(m.group('CtrlHex'), 16)
        
        ctrl = []
        ctrl.append((ctrlInst & 0x000000000001ffff) >> 0)
        ctrl.append((ctrlInst & 0x0000003fffe00000) >> 21)
        ctrl.append((ctrlInst & 0x07fffc0000000000) >> 42)
        
        reuse = []
        reuse.append((ctrlInst & 0x00000000001e0000) >> 17)
        reuse.append((ctrlInst & 0x000003c000000000) >> 38)
        reuse.append((ctrlInst & 0x7800000000000000) >> 59)
        
        return Flags(ctrl, reuse)
    else:
        raise ValueError("The input string is not SASS control line")

def processSassLine(line):
    """
    Extract SASS instruction from SASS code. 
    
    Args:
        line (str): a SASS line from cuobjdump
    
    Returns:
        Instruction: an Instruction object representing the SASS intruction 
    
    Raises:
        ValueError: If line is SASS line   
    """
    m = re.search("^\s+/\*(?P<num>[0-9a-f]+)\*/\s+(?P<pred>@!?(?P<predReg>P\d)\s+)?(?P<op>\w+)(?P<rest>[^;]*;)\s+/\* (?P<code>0x[0-9a-f]+)", line)
    if m:
        pred = m.group('pred') if m.group('pred') else ""
        
        return Instruction(
            int(m.group('num'), 16),
            pred,
            m.group('op'),
            m.group('op') + m.group('rest'),
            m.group('code')
        )
    else:
        raise ValueError("The input string is not SASS line: " + line)
 
def flagsToString(flag):
    """
    Convert SASS control flags to string.
    
    Equivalent to MaxAs::MaxAsGrammer::printCtrl
    
    Args:
        ctrl (str): a SASS control flags
        
    Returns:
        str: formatted SASS control flags
    """
    ctrlStall = (flag & 0x0000f) >> 0
    ctrlYield = (flag & 0x00010) >> 4
    ctrlWrtdb = (flag & 0x000e0) >> 5
    ctrlReadb = (flag & 0x00700) >> 8
    ctrlWatdb = (flag & 0x1f800) >> 11
    
    ctrlYield = '-' if ctrlYield == 1 else 'Y'
    ctrlWrtdb = '-' if ctrlWrtdb == 7 else ctrlWrtdb + 1
    ctrlReadb = '-' if ctrlReadb == 7 else ctrlReadb + 1
    ctrlWatdb = '--' if ctrlWatdb == 0 else "%02x" % (ctrlWatdb)
    
    return "%s:%s:%s:%s:%x" % (ctrlWatdb, ctrlReadb, ctrlWrtdb, ctrlYield, ctrlStall) 
    
def extract(args):
    """
    Generate Maxas compatible SASS file.
    """
    kernelName = args.kernel
    outputName = args.output
    outputFile = None
    kernel = None
    
    cubin = Cubin(args.input_file)

    if kernelName == None:
        kernelName = list(cubin.kernels.keys())[0]
        kernel = cubin.kernels[kernelName]
            
    sass = ""
    
    if args.cuobjdump:
        sassFile = open(args.cuobjdump, 'r') 
        sass = sassFile.readlines()
        sassFile.close()
    else:
        try:
            cuobjdumpSass = subprocess.check_output([CUDA_BIN_DIR+'cuobjdump','-arch', "sm_"+str(cubin.arch),'-sass','-fun', kernelName, args.input_file], universal_newlines=True)
            sass = cuobjdumpSass.split('\n')
        except FileNotFoundError:
            print("cuobjdump does not exist in this system. Please install CUDA toolkits before using this tool.")
            exit()
        except subprocess.CalledProcessError as err:
            print(err.cmd)
            exit()
    
    if outputName == None:
        outputFile = sys.stdout
    else:
        outputFile = open(outputName, 'w')
    
    outputFile.write("# Kernel: "+ kernelName +"\n" )
    outputFile.write("# Arch: sm_"+ str(cubin.arch) +"\n" )
    outputFile.write("# InsCnt: " +"\n" )
    outputFile.write("# RegCnt: "+ str(kernel['RegisterCount']) +"\n" )
    outputFile.write("# SharedSize: "+ str(kernel['SharedSize']) +"\n" )
    outputFile.write("# BarCnt: "+ str(kernel['BarrierCount']) +"\n" )
    outputFile.write("# Params(" + str(kernel['ParameterCount']) +"):\n#\tord:addr:size:align\n")
    for param in kernel['Parameters']:
        outputFile.write("#\t" + param + "\n")
    outputFile.write("# Instructions:\n\n")
    
    
    paramsMap = {}
    constants = {
        'blockDimX' : 'c[0x0][0x8]',
        'blockDimY' : 'c[0x0][0xc]',
        'blockDimZ' : 'c[0x0][0x10]',
        'gridDimX' : 'c[0x0][0x14]',
        'gridDimY' : 'c[0x0][0x18]',
        'gridDimZ' : 'c[0x0][0x1c]',
    }
    
    constants = OrderedDict(sorted(constants.items(), key=lambda t: t[0]))
    
    outputFile.write("<CONSTANT_MAPPING>\n")
    for const in constants:
        #outputFile.write("\t" + const + " : " + constants[const] + "\n")
        paramsMap[constants[const]] = const
    
    for param in kernel['Parameters']:
        ord, offset, size, align = param.split(':')
        size = int(size)
        ord = int(ord)
        offset = int(offset, 16)
        
        if size > 4:
            num = 0
            while size > 0:
                p = "param_%d[%d]" % (ord, num)
                c = "c[0x0][0x%x]" % (offset)
                paramsMap[c] = p
                constants[p] = c
                #outputFile.write("\t" + p + " : " + c +"\n")
                size -= 4
                offset += 4
                num += 1
        else:
            p = "param_%d" % (ord)
            c = "c[0x0][0x%x]" % (offset)
            paramsMap[c] = p
            constants[p] = c
            #outputFile.write("\t" + p + " : " + c +"\n")
    
    json.dump(constants, outputFile, indent=4)
            
    outputFile.write("\n</CONSTANT_MAPPING>\n\n")
    
    linePtr = 0
    
    labels = {}
    labelNum = 1
    instSet = []
    
    while linePtr < len(sass):
        line = sass[linePtr]
        linePtr += 1
        
        if not isSassCtrlLine(line):
            continue
        
        flags = processSassCtrlFlags(line)
        
        for flag in flags.ctrl:
            line = sass[linePtr]
            linePtr += 1
            
            inst = processSassLine(line)

            # Convert branch/jump/call addresses to labels
            if inst.opcode in JUMP_OPS and re.search("(?P<target>0x[0-9a-f]+)", inst.inst):
                m = re.search("(?P<target>0x[0-9a-f]+)", inst.inst)
                target = int(m.group('target'), 16)
                
                # Skip the final BRA and stop processing the file
                if inst.opcode == 'BRA' and (target == inst.line or target == inst.line - 8):
                    linePtr = len(sass)
                    break
                # Check to see if we've already generated a label for this target address
                if not labels.get(target):
                    # Generate a label name and cache it
                    labels[target] = "TARGET" + str(labelNum)
                    labelNum += 1
                
                inst.inst = re.sub("(0x[0-9a-f]+)", labels[target], inst.inst)
                
            constMatch = re.search("(c\[0x0\])\s*(\[0x[0-9a-f]+\])", inst.inst)   
            if constMatch and (constMatch.group(1) + constMatch.group(2)) in paramsMap.keys():         
                inst.inst = re.sub(
                    "(c\[0x0\])\s*(\[0x[0-9a-f]+\])", 
                    paramsMap[constMatch.group(1) + constMatch.group(2)] , 
                    inst.inst)
            
            inst.flag = flagsToString(flag)
            instSet.append(inst)
            
    for inst in instSet:
        if labels.get(inst.line):
            outputFile.write(labels[inst.line] + ":\n")
        outputFile.write("%s %5s%s\n" % (inst.flag, inst.pred, inst.inst))
        
  