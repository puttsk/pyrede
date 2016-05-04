import re
import pprint
from collections import namedtuple

REL_OFFSETS = ['BRA', 'SSY', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

def isSassCtrlLine(line):
    """
    Check if the input line is control instruction
    
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
    Extract control and reuse flags from SASS control instruction
    
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
    Extract SASS instruction from SASS code
    
    Args:
        line (str): a SASS line from cuobjdump
    
    Returns:
        {
            'num': 
            'pred': 
            'op':
            'ins':
            'inst':
            'code':
        } 
    
    Raises:
        ValueError: If line is SASS line   
    """
    m = re.search("^\s+/\*(?P<num>[0-9a-f]+)\*/\s+(?P<pred>@!?(?P<predReg>P\d)\s+)?(?P<op>\w+)(?P<rest>[^;]*;)\s+/\* (?P<code>0x[0-9a-f]+)", line)
    
    if m:
        Inst = namedtuple('Inst', ['num', 'pred', 'op', 'ins', 'inst', 'code'])
        pred = m.group('pred') if m.group('pred') else ""
               
        return Inst(
            int(m.group('num'), 16),
            pred,
            m.group('op'),
            m.group('op') + m.group('rest'),
            pred + str(m.group('op')) + str(m.group('rest')),
            m.group('code')
        )
    else:
        raise ValueError("The input string is not SASS line")
    
def extract(sass, outputFile, params):
    """
    Generate Maxas compatible SASS file
    """
    paramsMap = {}
    constants = {
        'blockDimX' : 'c[0x0][0x8]',
        'blockDimY' : 'c[0x0][0xc]',
        'blockDimZ' : 'c[0x0][0x10]',
        'gridDimX' : 'c[0x0][0x14]',
        'gridDimY' : 'c[0x0][0x18]',
        'gridDimZ' : 'c[0x0][0x1c]',
    }
    
    outputFile.write("<CONSTANT_MAPPING>\n")
    for const in constants:
        outputFile.write("\t" + const + " : " + constants[const] + "\n")
        paramsMap[constants[const]] = const
    outputFile.write("\n")
    
    for param in params:
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
                outputFile.write("\t" + p + " : " + c +"\n")
                size -= 4
                offset += 4
                num += 1
        else:
            p = "param_%d" % (ord)
            c = "c[0x0][0x%x]" % (offset)
            paramsMap[c] = p
            outputFile.write("\t" + p + " : " + c +"\n")
    outputFile.write("</CONSTANT_MAPPING>\n\n")
    
    sass = sass.split('\n')
    linePtr = 0
    
    labels = {}
    labelNum = 1
    
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
            if inst.op in JUMP_OPS and re.search("(?P<target>0x[0-9a-f]+)", inst.ins):
                m = re.search("(?P<target>0x[0-9a-f]+)", inst.ins)
                target = m.group('target')
                
                # Skip the final BRA and stop processing the file
                if inst.op == 'BRA' and (int(target, 16) == inst.num or int(target, 16) == inst.num - 8):
                    linePtr = len(sass)
                    break
                # Check to see if we've already generated a label for this target address
                if not labels.get(target):
                    # Generate a label name and cache it
                    labels[target] = "TARGET" + str(labelNum)
                    labelNum += 1
                
                inst = inst._replace(ins = re.sub("(0x[0-9a-f]+)", labels[target], inst.ins))
                
            constMatch = re.search("(c\[0x0\])\s*(\[0x[0-9a-f]+\])", inst.ins)   
            if constMatch and (constMatch.group(1) + constMatch.group(2)) in paramsMap.keys():         
                inst = inst._replace(ins = re.sub(
                    "(c\[0x0\])\s*(\[0x[0-9a-f]+\])", 
                    paramsMap[constMatch.group(1) + constMatch.group(2)] , 
                    inst.ins))
            print(inst.ins)
                
            
            