import sys
import binascii
import struct
import pprint

DEBUG = False

#ASSUME: the CUBIN file use system encoding (little/big endian)

# CUBIN 32-bit ELF header format
Elf32_Hdr = [
    '=',
    '4s',  # 4-bytes magic
    'B',  # fileClass
    'B',  # encoding
    'B',  # fileVersion
    '9s',  # 9-bytes padding
    'H',  # type
    'H',  # machine
    'L',  # version
    'L',  # entry
    'L',  # phOffset
    'L',  # shOffset
    'L',  # flags
    'H',  # ehSize
    'H',  # phEntSize
    'H',  # phNum
    'H',  # shEntSize
    'H',  # shNum
    'H'  # shStrIndx
]

# CUBIN 64-bit ELF header format
Elf64_Hdr = [
    '=',
    '4s',  # 4-bytes magic
    'B',  # fileClass
    'B',  # encoding
    'B',  # fileVersion
    '9s',  # 9-bytes padding
    'H',  # type
    'H',  # machine
    'L',  # version
    'Q',  # entry
    'Q',  # phOffset
    'Q',  # shOffset
    'L',  # flags
    'H',  # ehSize
    'H',  # phEntSize
    'H',  # phNum
    'H',  # shEntSize
    'H',  # shNum
    'H'  # shStrIndx
]

# CUBIN ELF header field
Elf_Hdr_Field = [
    'magic',
    'fileClass',
    'encoding',
    'fileVersion',
    'padding',  # 9-bytes padding
    'type',
    'machine',
    'version',
    'entry',
    'phOffset',
    'shOffset',
    'flags',
    'ehSize',
    'phEntSize',
    'phNum',
    'shEntSize',
    'shNum',
    'shStrIndx',
]

# CUBIN 32-bit ELF Program header format
Elf32_PrgHdr = [
    '=',
    'L',#   type
    'L',#   offset
    'L',#   vaddr
    'L',#   paddr
    'L',#   fileSize
    'L',#   memSize
    'L',#   flags
    'L',#   align
]

# CUBIN 64-bit ELF Program header format
Elf64_PrgHdr = [
    '=',
    'L',#   type
    'L',#   flags
    'Q',#   offset
    'Q',#   vaddr
    'Q',#   paddr
    'Q',#   fileSize
    'Q',#   memSize
    'Q',#   align
]

# CUBIN ELF Program header field
Elf_PrgHdr_Field = [
    'type',
    'flags',
    'offset',
    'vaddr',
    'paddr',
    'fileSize',
    'memSize',
    'align'
]

# CUBIN 32-bit ELF Section header format
Elf32_SecHdr = [
    '=',
    'L',#   name
    'L',#   type
    'L',#   flags
    'L',#   addr
    'L',#   offset
    'L',#   size
    'L',#   link
    'L',#   info
    'L',#   align
    'L',#   entSize
]

# CUBIN 64-bit ELF Section header format
Elf64_SecHdr = [
    '=',
    'L',#   name
    'L',#   type
    'Q',#   flags
    'Q',#  addr
    'Q',#   offset
    'Q',#   size
    'L',#   link
    'L',#   info
    'Q',#   align
    'Q',#   entSize
]

# CUBIN ELF Section header field
Elf_SecHdr_Field = [
    'name',
    'type',
    'flags',
    'addr',
    'offset',
    'size',
    'link',
    'info',
    'align',
    'entSize',
]

# CUBIN 32-bit ELF Section entry format
Elf32_SymEnt = [
    '=',
    'L',#   name
    'L',#   value
    'L',#   size
    'B',#   info
    'B',#   other
    'H',#   shIndx
]

# CUBIN 64-bit ELF Section header format
Elf64_SymEnt = [
    '=',
    'L',#   name
    'B',#   info
    'B',#   other
    'H',#   shIndx
    'Q',#   value
    'Q',#   size
]

# CUBIN ELF Section entry field
Elf_SymEnt_Field = [
    'name',
    'info',
    'other',
    'shIndx',
    'value',
    'size',
]

SymbolBind = [
    'LOCAL',
    'GLOBAL',
    'WEAK'
]

# Base class for CUBIN file
class Cubin():

    def __init__(self, fileName):
        try:
            self.fileName = fileName
            cubinFile = open(fileName, 'rb')

            # Try reading header assuming 32-bit header
            headerBin = cubinFile.read(0x34)
            headerData = list(struct.unpack("".join(Elf32_Hdr), headerBin))

            self.section = {}
            self.kernels = {}
            self.symbols = {}
            self.elfHeader = dict(zip(Elf_Hdr_Field, headerData))

            # Check if the fileClass is 32-bit or 64-bit
            # 1: 32-bit    2: 64-bit
            if self.elfHeader['fileClass'] == 2:
                cubinFile.seek(0)
                headerBin = cubinFile.read(0x46)
                headerData = list(struct.unpack_from("".join(Elf64_Hdr), headerBin))
                self.elfHeader = dict(zip(Elf_Hdr_Field, headerData))

                cubinFileClass = 64
            else:
                cubinFileClass = 32

            self.arch = self.elfHeader['flags'] & 0xFF
            self.addressSize = 64 if self.elfHeader['flags'] & 0x400 else 32

            if DEBUG:
                pprint.pprint("ELF header")
                pprint.pprint(self.elfHeader)
                pprint.pprint("==============")
            
            # Read Program header
            cubinFile.seek(self.elfHeader['phOffset'], 0)
            self.programHeader = []
            for i in range(0, self.elfHeader['phNum']):
                prgHeaderBin = cubinFile.read(self.elfHeader['phEntSize'])
                prgHeader = ''
                
                if cubinFileClass == 64:
                    prgHeader = list(struct.unpack("".join(Elf64_PrgHdr), prgHeaderBin))
                else:
                    prgHeader = list(struct.unpack("".join(Elf32_PrgHdr), prgHeaderBin))
                self.programHeader.append(dict(zip(Elf_PrgHdr_Field, prgHeader)))

            if DEBUG:
                pprint.pprint("Program header")
                pprint.pprint(self.programHeader)
                pprint.pprint("==============")
            
             # Read Section header    
            cubinFile.seek(self.elfHeader['shOffset'], 0)
            self.sectionHeader = []
            for i in range(0, self.elfHeader['shNum']):
                sectionHeaderBin = cubinFile.read(self.elfHeader['shEntSize'])
                sectionHeader = ''
                
                if cubinFileClass == 64:
                    sectionHeader = list(struct.unpack_from("".join(Elf64_SecHdr), sectionHeaderBin))
                else:
                    sectionHeader = list(struct.unpack_from("".join(Elf32_SecHdr), sectionHeaderBin))
                self.sectionHeader.append(dict(zip(Elf_SecHdr_Field, sectionHeader)))
            
            # Read Section data
            for section in self.sectionHeader:
                data = ''
                
                # Skip sections with no data (type NULL or NOBITS)
                if section['size'] and section['type'] != 8:
                    cubinFile.seek(section['offset'], 0)
                    data = cubinFile.read(section['size'])
                
                # Convert string tables to maps
                # ASSUME: ASCII encoding
                if section['type'] == 3: #STRTAB
                    section['StrTab'] = {}
                    idx = 0
                    for strData in data.decode('ascii').split('\0'):
                        section['StrTab'][idx] = strData
                        idx += 1 + len(strData)
                # Read Symbol data
                if section['type'] == 2: #SIMTAB
                    offset = 0
                    section['SymTab'] = []
                    while offset < section['size']:
                        symbolEntryData = ''
                        if cubinFileClass == 64:
                            symbolEntryData = list(struct.unpack("".join(Elf64_SymEnt), data[offset:offset+section['entSize']]))
                        else:
                            symbolEntryData = list(struct.unpack("".join(Elf32_SymEnt), data[offset:offset+section['entSize']]))
                        section['SymTab'].append(dict(zip(Elf_SymEnt_Field, symbolEntryData)))
                        offset += section['entSize']      
                
                # Cache raw data for further processing and writing
                section['Data'] = data
            
            cubinFile.close()
            
            # Update section headers with their names.  Map names directly to headers.
            stringTable = self.sectionHeader[self.elfHeader['shStrIndx']]['StrTab']
            for secHeader in self.sectionHeader:
                secHeader['Name'] =  stringTable[secHeader['name']]
                self.section[secHeader['Name']] = secHeader
            
            # Update symbols with their names
            # For the Global functions, extract kernel meta data
            # Populate the kernel hash
            stringTable = self.section['.strtab']['StrTab']
            for symbolEntry in self.section['.symtab']['SymTab']:
                symbolEntry['Name'] = stringTable[symbolEntry['name']]
                
                #Attach symbol to section
                sectionHeader = self.sectionHeader[symbolEntry['shIndx']]
                sectionHeader['SymbolEnt'] = symbolEntry
            
                # Look for symbols with FUNC tag
                if symbolEntry['info'] & 0x0f == 0x02:
                    # Create a hash of kernels for output
                    self.kernels[symbolEntry['Name']] = sectionHeader
                    kernelSection = self.kernels[symbolEntry['Name']]
                    
                    # Extract local/global/weak binding info
                    kernelSection['Linkage'] = SymbolBind[(symbolEntry['info'] & 0xf0) >> 4] 
                    
                    # Extract the kernel instructions
                    kernelSection['KernelData'] = list(struct.iter_unpack('=Q', kernelSection['Data'])) 
                    kernelSection['KernelData'] = list(map(lambda x: x[0], kernelSection['KernelData']))
                    
                    # Extract the max barrier resource identifier used and add 1. Should be 0-16.
                    # If a register is used as a barrier resource id, then this value is the max of 16.
                    kernelSection['BarrierCount'] = (kernelSection['flags'] & 0x01f00000) >> 20
                    
                    # Extract the number of allocated registers for this kernel.
                    kernelSection['RegisterCount'] = (kernelSection['info'] & 0xff000000) >> 24
                    
                    # Extract the size of shared memory this kernel uses.
                    kernelSection['SharedSection'] = self.section.get('.nv.shared.' + symbolEntry['Name'])
                    kernelSection['SharedSize'] =  0
                    if kernelSection['SharedSection']:
                        kernelSection['SharedSize'] = kernelSection['SharedSection']['size']

                    # Attach constant0 section
                    kernelSection['ConstantSection'] = self.section.get('.nv.constant0.' + symbolEntry['Name'])
                    
                    # Extract the kernel parameter data.
                    kernelSection['ParameterSection'] = self.section.get('.nv.info.' + symbolEntry['Name'])
                    if kernelSection['ParameterSection']:
                        paramSec = kernelSection['ParameterSection']
                        
                        #Extract raw parameter data
                        data = struct.iter_unpack('=L', paramSec['Data'])
                        data = list(map(lambda x: x[0], data))
                        
                        paramSec['ParameterData'] = data
                        paramSec['ParameterHex'] = list(map(lambda x: ("0x%08x")%x, data))
                        
                        #Find the first parameter delimiter
                        idx = 0
                        while idx < len(data) and data[idx] != 0x00080a04:
                            idx += 1
                        
                        first = data[idx+2] & 0xFFFF
                        idx += 4
                        
                        params = []
                        
                        while idx < len(data) and data[idx] == 0x000c1704:
                            # Get the ordinal, offset, size and pointer alignment for each param
                            ordinal = data[idx+2] & 0xFFFF
                            offset = "0x%02x" % (first + (data[idx+2] >> 16))
                            psize = data[idx + 3] >> 18
                            align = 1 << (data[idx+3] & 0x3FF) if (data[idx+3] & 0x400) else 0
                            params.insert(0, (str(ordinal) + ":" + offset + ":" + str(psize) + ":" + str(align)))
                            idx += 4
                        
                        staticParams = data[0:idx]
                        
                        maxRegCount = 0
                        ctaidOffsets = []
                        ctaidzUsed = 0
                        exitOffsets = []
                        reqntid = []
                        maxntid = []
                        stackSize = []
                        
                        while idx < len(data):
                            code = data[idx] & 0xFFFF
                            size = data[idx] >> 16
                            idx += 1
                            
                            # EIATTR_MAXREG_COUNT
                            if code == 0x1B03:
                                maxRegCount = size
                            # EIATTR_S2RCTAID_INSTR_OFFSETS    
                            elif code == 0x1D04:
                                while size > 0:
                                    ctaidOffsets.append(data[idx])
                                    idx += 1
                                    size -= 4
                            # EIATTR_EXIT_INSTR_OFFSET
                            elif code == 0x1C04:
                                while size > 0:
                                    exitOffsets.append(data[idx])
                                    idx += 1
                                    size -= 4
                            # EIATTR_CTAIDZ_USED
                            elif code == 0x0401:
                                ctaidzUsed = 1
                            # EIATTR_REQNTID
                            elif code == 0x1004:
                                while size > 0:
                                    reqntid.append(data[idx])
                                    idx += 1
                                    size -= 4
                            # EIATTR_MAX_THREADS
                            elif code == 0x0504:
                                while size > 0:
                                    maxntid.append(data[idx])
                                    idx += 1
                                    size -= 4
                            # EIATTR_CRS_STACK_SIZE
                            elif code == 0x1E04:
                                 while size > 0:
                                    stackSize.append(data[idx])
                                    idx += 1
                                    size -= 4
                            else:
                                print("Unknown Code 0x%02x (size:%d)" % (code, size))
                                
                        kernelSection['Parameters'] = params
                        kernelSection['ParameterCount'] = len(params)   
                        
                        paramSec['StaticParams'] = staticParams
                        paramSec['MAXREG_COUNT'] = maxRegCount
                        paramSec['ExitOffsets']  = exitOffsets
                        paramSec['CTAIDOffsets'] = ctaidOffsets
                        paramSec['CTAIDZUsed']   = ctaidzUsed
                        paramSec['REQNTID']      = reqntid
                        paramSec['MAXNTID']      = maxntid
                        paramSec['STACKSIZE']    = stackSize
                
                # Note GLOBALs found in this cubin
                elif symbolEntry['info'] & 0x0f == 0x02:
                    self.symbols[symbolEntry['Name']] = symbolEntry 
                                                                                                 
            if DEBUG:
                pprint.pprint("Section header")
                pprint.pprint(self.sectionHeader)
                pprint.pprint("==============")
                pprint.pprint(self.kernels)
        except struct.error:
            print("The input file is not Cubin: " + self.fileName)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
    
    def printInfo(self):
        print("%s: \n\tarch: sm_%d \n\tmachine: %d bit \n\taddress_size: %d bit\n" % 
            (   self.fileName, 
                self.arch, 
                self.addressSize, 
                self.addressSize))
        for kernel in self.kernels:
            self.__printKernelInfo(self.kernels[kernel])
        for symbol in self.symbols:
            print("Symbol: " + symbol + "\n")
    
    def __printKernelInfo(self, kernel):
        print("Kernel: " + kernel['Name'])
        print("\tLinkage: %s \n\tParams: %d \n\tSize: %d \n\tRegisters: %d \n\tSharedMem: %d \n\tBarriers: %d" % 
        (   kernel['Linkage'], 
            kernel['ParameterCount'], 
            kernel['size'],
            kernel['RegisterCount'],
            kernel['SharedSize'],
            kernel['BarrierCount']))
        