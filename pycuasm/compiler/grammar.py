from collections import namedtuple

# allow_pred : True if predicate clause is allowed for this instruction 
# reg_store  : True if the instruction stores its result in the first register
Grammar = namedtuple('Grammar', ['type', 'code', 'rule', 'allow_pred', 'reg_store', 'integer_inst','float_inst', 'is_64']) 
Grammar.__new__.__defaults__ = (None, None, None, False, False, False, False, False)

#Taken from MaxAs
SASS_GRAMMARS = {    
    #Floating Point Instructions
    'FADD'     : Grammar('x32', 0x5c58000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$FADD'f'z$rnd'sa' $r0, $r8, $fcr20"),
    'FADD32I'  : Grammar('x32', 0x0800000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$FADD32I'f'z $r0, $r8, $f20w32"),
    'FCHK'     : Grammar('x32', 0x5c88000000000000, allow_pred = True, reg_store = False, float_inst = True, rule = r"^$FCHK\.DIVIDE $p0, $r8, $r20"), #Partial?
    'FCMP'     : Grammar('cmp', 0x5ba0000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$FCMP$fcmp'f'z $r0, $r8, $fcr20, $r39"),
    'FFMA'     : Grammar('x32', 0x5980000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = [r"^$FFMA'f'z$rnd'sa' $r0, $r8, $fcr20, $r39",r"^$FFMA'f'z$rnd'sa' $r0, $r8, $r39s20, $c20s39"]),
    'FMNMX'    : Grammar('shft',0x5c60000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$FMNMX'f'z $r0, $r8, $fcr20, $p39"),
    'FMUL'     : Grammar('x32', 0x5c68000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$FMUL'f'z$rnd'sa' $r0, $r8, $fcr20"),
    'FMUL32I'  : Grammar('x32', 0x1e00000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$FMUL32I'f'z $r0, $r8, $f20w32"),
    'FSET'     : Grammar('shft',0x5800000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$FSET$fcmp'f'z$bool $r0, $r8, $fcr20, $p39"),
    'FSETP'    : Grammar('cmp', 0x5bb0000000000000, allow_pred = True, reg_store = False, float_inst = True, rule = r"^$FSETP$fcmp'f'z$bool $p3, $p0, $r8, $fcr20, $p39"),
    'MUFU'     : Grammar('qtr', 0x5080000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$MUFU$func $r0, $r8"),
    'RRO'      : Grammar('rro', 0x5c90000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$RRO$rro $r0, $r20"),
    'DADD'     : Grammar('x64', 0x5c70000000000000, allow_pred = True, reg_store = True, float_inst = True, is_64 = True, rule = r"^$DADD$rnd $r0, $r8, $dr20"),
    'DFMA'     : Grammar('x64', 0x5b70000000000000, allow_pred = True, reg_store = True, float_inst = True, is_64 = True, rule = r"^$DFMA$rnd $r0, $r8, $dr20, $r39"),
    'DMNMX'    : Grammar('cmp', 0x5c50000000000000, allow_pred = True, reg_store = True, float_inst = True, is_64 = True, rule = r"^$DMNMX $r0, $r8, $dr20, $p39"),
    'DMUL'     : Grammar('x64', 0x5c80000000000000, allow_pred = True, reg_store = True, float_inst = True, is_64 = True, rule = r"^$DMUL$rnd $r0, $r8, $dr20"),
    'DSET'     : Grammar('cmp', 0x5900000000000000, allow_pred = True, reg_store = True, float_inst = True, is_64 = True, rule = r"^$DSET$fcmp$bool $r0, $r8, $dr20, $p39"),
    'DSETP'    : Grammar('cmp', 0x5b80000000000000, allow_pred = True, reg_store = False, float_inst = True, is_64 = True, rule = r"^$DSETP$fcmp$bool $p3, $p0, $r8, $dr20, $p39"),
    'FSWZADD'  : Grammar('x32', 0x0000000000000000, allow_pred = True, reg_store = True, float_inst = True, rule = r"^$FSWZADD[^]*"), #TODO
    'HADD2'     : Grammar('x32', 0x5d10000000000000, allow_pred = True, reg_store = True, rule = r"^$HADD2'f'z $r0, $r8, $r20"),
    'HMUL2'     : Grammar('x32', 0x5d08000000000000, allow_pred = True, reg_store = True, rule = r"^$HMUL2'f'z $r0, $r8, $r20"),
    'HFMA2'     : Grammar('x32', 0x5d00000000000000, allow_pred = True, reg_store = True, rule = r"^$HFMA2'f'z $r0, $r8, $r20, $r39"),
    'HSETP2'    : Grammar('cmp', 0x5d20000000000000, allow_pred = True, reg_store = False, rule = r"^$HSETP2$fcmp$bool $p3, $p0, $r8, $fcr20, $p39"), #Partial

    #Integer Instructions
    'BFE'       : Grammar('shft', 0x5c01000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$BFE$u32 $r0, $r8, $icr20"),
    'BFI'       : Grammar('shft', 0x5bf0000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$BFI $r0, $r8, $ir20, $cr39"),
    'FLO'       : Grammar('s2r',  0x5c30000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$FLO\.U32 $r0, $icr20"),
    'IADD'      : Grammar('x32',  0x5c10000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$IADD'sa'$X $r0cc, $r8, $icr20"),
    'IADD32I'   : Grammar('x32',  0x1c00000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$IADD32I$X $r0cc, $r8, $i20w32"),
    'IADD3'     : Grammar('x32',  0x5cc0000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$IADD3$add3 $r0cc, $r8, $icr20, $r39"),
    'ICMP'      : Grammar('cmp',  0x5b41000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$ICMP$icmp$u32 $r0, $r8, $icr20, $r39"),
    'IMNMX'     : Grammar('shft', 0x5c21000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$IMNMX$u32$hilo $r0cc, $r8, $icr20, $p39"),
    'ISET'      : Grammar('shft', 0x5b51000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$ISET$icmp$u32$X$bool $r0, $r8, $icr20, $p39"),
    'ISETP'     : Grammar('cmp',  0x5b61000000000000, allow_pred = True, reg_store = False, integer_inst = True, rule = r"^$ISETP$icmp$u32$X$bool $p3, $p0, $r8, $icr20, $p39"),
    'ISCADD'    : Grammar('shft', 0x5c18000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$ISCADD $r0, $r8, $icr20, $i39w8"),
    'ISCADD32I' : Grammar('shft', 0x1400000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$ISCADD32I $r0, $r8, $i20w32, $i53w5"),
# TODO
#    'LEA'       : Grammar('cmp',  0x5bd0000000000000, allow_pred = True, reg_store = False, rule = r"^$LEA $p48, $r0cc, $r8, $icr20"),
#                   Grammar('shft', 0x5bd7000000000000, allow_pred = True, reg_store = True, rule = r"^$LEA $r0cc, $r8, $icr20, $i39w8"),
#                   Grammar('shft', 0x5bdf004000000000, allow_pred = True, reg_store = True, rule = r"^$LEA\.HI$X $r0cc, $r8, $r20, $r39, $i28w8"),
#                   Grammar('shft', 0x0a07000000000000, allow_pred = True, reg_store = True, rule = r"^$LEA\.HI$X $r0cc, $r8, $c20, $r39, $i51w5"),
#                 ],
    'LOP'       : Grammar('x32',  0x5c40000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$LOP$bool$lopz $r0, $r8, (?<INV>~)?$icr20(?<INV>\.INV)?"),
    'LOP32I'    : Grammar('x32',  0x0400000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$LOP32I$bool $r0, $r8, $i20w32"),
    'LOP3'      : Grammar('x32',  0x5be7000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = [r"^$LOP3\.LUT $r0, $r8, $r20, $r39, $i28w8",r"^$LOP3\.LUT $r0, $r8, $i20, $r39, $i48w8"]),
    'POPC'      : Grammar('s2r',  0x5c08000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$POPC $r0, $r20"),
    'SHF'       : Grammar('shft', 0x5bf8000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = [ r"^$SHF\.L$shf $r0, $r8, $ir20, $r39", r"^$SHF\.R$shf $r0, $r8, $ir20, $r39",]),
    'SHL'       : Grammar('shft', 0x5c48000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$SHL(?<W>\.W)? $r0, $r8, $icr20"),
    'SHR'       : Grammar('shft', 0x5c29000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = r"^$SHR$u32 $r0, $r8, $icr20"),
    'XMAD'      : Grammar('x32',  0x5b00000000000000, allow_pred = True, reg_store = True, integer_inst = True, rule = [ r"^$XMAD$xmad $r0cc, $r8, $ir20, $r39", r"^$XMAD$xmad $r0cc, $r8, $r39s20, $c20s39", r"^$XMAD$xmadc $r0cc, $r8, $c20x, $r39"] ),
    # XMAD replaces these
    'IMAD'      : Grammar('x32',  0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$IMAD[^]*"), #TODO
    'IMADSP'    : Grammar('x32',  0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$IMADSP[^]*"), #TODO
    'IMUL'      : Grammar('x32',  0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$IMUL[^]*"), #TODO

    #Conversion Instructions
    'F2F' : Grammar('qtr', 0x5ca8000000000000, allow_pred = True, reg_store = True, rule = r"^$F2F'f'z$x2x$rnd$round'sa' $r0, $cr20"),
    'F2I' : Grammar('qtr', 0x5cb0000000000000, allow_pred = True, reg_store = True, rule = r"^$F2I'f'z$x2x$round $r0, $cr20"),
    'I2F' : Grammar('qtr', 0x5cb8000000000000, allow_pred = True, reg_store = True, rule = r"^$I2F$x2x$rnd $r0, $cr20"),
    'I2I' : Grammar('qtr', 0x5ce0000000000000, allow_pred = True, reg_store = True, rule = r"^$I2I$x2x'sa' $r0, $cr20"),

    #Movement Instructions
    'MOV'    : Grammar('x32', 0x5c98078000000000, allow_pred = True, reg_store = True, rule = r"^$MOV $r0, $icr20"),
    'MOV32I' : Grammar('x32', 0x010000000000f000, allow_pred = True, reg_store = True, rule = r"^$MOV32I $r0, (?:$i20w32|$f20w32)"),
    'PRMT'   : Grammar('x32', 0x5bc0000000000000, allow_pred = True, reg_store = True, rule = r"^$PRMT'prm' $r0, $r8, $icr20, $cr39"),
    'SEL'    : Grammar('x32', 0x5ca0000000000000, allow_pred = True, reg_store = True, rule = r"^$SEL $r0, $r8, $icr20, $p39"),
    'SHFL'   : Grammar('smem',0xef10000000000000, allow_pred = True, reg_store = False, rule = r"^$SHFL$shfl $p48, $r0, $r8, (?:$i20w8|$r20), (?:$i34w13|$r39)"),

    #Predicate/CC Instructions
    'PSET'   : Grammar('cmp', 0x5088000000000000, allow_pred = True, reg_store = True, rule = r"^$PSET$bool2$bool $r0, $p12, $p29, $p39"),
    'PSETP'  : Grammar('cmp', 0x5090000000000000, allow_pred = True, reg_store = False, rule = r"^$PSETP$bool2$bool $p3, $p0, $p12, $p29, $p39"),
    'CSET'   : Grammar('x32', 0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$CSET[^]*"), #TODO
    'CSETP'  : Grammar('x32', 0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$CSETP[^]*"), #TODO
    'P2R'    : Grammar('x32', 0x38e8000000000000, allow_pred = True, reg_store = True, rule = r"^$P2R $r0, PR, $r8, $i20w7"),
    'R2P'    : Grammar('shft',0x38f0000000000000, allow_pred = True, reg_store = True, rule = r"^$R2P PR, $r8, $i20w7"),

    #Texture Instructions
    # Handle the commonly used 1D texture functions.. but save the others for later
    'TLD'    : Grammar('gmem',0xdd38000000000000, allow_pred = True, reg_store = True, rule = r"^$TLD\.B\.LZ\.$tld $r0, $r8, $r20, $hex, \dD, $i31w4"), #Partial
    'TLDS'   : Grammar('gmem',0xda0000000ff00000, allow_pred = True, reg_store = True, rule = r"^$TLDS\.LZ\.$tld $r28, $r0, $r8, $i36w20, \dD, $chnls",), #Partial
    'TEX'    : Grammar('gmem',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$TEX[^]*"), #TODO
    'TLD4'   : Grammar('gmem',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$TLD4[^]*"), #TODO
    'TXQ'    : Grammar('gmem',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$TXQ[^]*"), #TODO
    'TEXS'   : Grammar('gmem',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$TEXS[^]*"), #TODO
    'TLD4S'  : Grammar('gmem',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$TLD4S[^]*"), #TODO

    #Compute Load/Store Instructions
    'LD'     : Grammar('gmem',0x8000000000000000, allow_pred = True, reg_store = True, rule = r"^$LD$memCache$mem'type' $r0, $addr, $p58"),
    'ST'     : Grammar('gmem',0xa000000000000000, allow_pred = True, reg_store = False, rule = r"^$ST$memCache$mem'type' $addr, $r0, $p58"),
    'LDG'    : Grammar('gmem',0xeed0000000000000, allow_pred = True, reg_store = True, rule = r"^$LDG$memCache$mem'type' $r0, $addr"),
    'STG'    : Grammar('gmem',0xeed8000000000000, allow_pred = True, reg_store = False, rule = r"^$STG$memCache$mem'type' $addr, $r0"),
    'LDS'    : Grammar('smem',0xef48000000000000, allow_pred = True, reg_store = True, rule = r"^$LDS$memCache$mem'type' $r0, $addr"),
    'STS'    : Grammar('smem',0xef58000000000000, allow_pred = True, reg_store = False, rule = r"^$STS$memCache$mem'type' $addr, $r0"),
    'LDL'    : Grammar('gmem',0xef40000000000000, allow_pred = True, reg_store = True, rule = r"^$LDL$memCache$mem'type' $r0, $addr"),
    'STL'    : Grammar('gmem',0xef50000000000000, allow_pred = True, reg_store = False, rule = r"^$STL$memCache$mem'type' $addr, $r0"),
    'LDC'    : Grammar('gmem',0xef90000000000000, allow_pred = True, reg_store = True, rule = r"^$LDC$memCache$mem'type' $r0, $ldc"),
    # Note for ATOM(S).CAS operations the last register needs to be in sequence with the second to last (as it's not en'code'd).
    'ATOM'   : Grammar('gmem',0xed00000000000000, allow_pred = True, reg_store = True, rule = r"^$ATOM'a'om $r0, $addr2, $r20(?:, $r39a)?"),
    'ATOMS'  : Grammar('smem',0xec00000000000000, allow_pred = True, reg_store = True, rule = r"^$ATOMS'a'om $r0, $addr2, $r20(?:, $r39a)?"),
    'RED'    : Grammar('gmem',0xebf8000000000000, allow_pred = True, reg_store = False, rule = r"^$RED'a'om $addr2, $r0"),
    'CCTL'   : Grammar('x32', 0x5c88000000000000, allow_pred = True, reg_store = True, rule = r"^$CCTL[^]*"), #TODO
    'CCTLL'  : Grammar('x32', 0x5c88000000000000, allow_pred = True, reg_store = True, rule = r"^$CCTLL[^]*"), #TODO
    'CCTLT'  : Grammar('x32', 0x5c88000000000000, allow_pred = True, reg_store = True, rule = r"^$CCTLT[^]*"), #TODO

    #Surface Memory Instructions (haven't gotten to these yet..)
    'SUATOM' : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$SUATOM[^]*"), #TODO
    'SULD'   : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$SULD[^]*"), #TODO
    'SURED'  : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$SURED[^]*"), #TODO
    'SUST'   : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = True, rule = r"^$SUST[^]*"), #TODO

    #Control Instructions
    'BRA'    : Grammar('x32',0xe24000000000000f, allow_pred = True, reg_store = False, rule = [ r"^$BRA(?<U>\.U)? $i20w24", r"^$BRA(?<U>\.U)? CC\.EQ, $i20w24",]),
    'BRX'    : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = False, rule = r"^$BRX[^]*"), #TODO
    'JMP'    : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = False, rule = r"^$JMP[^]*"), #TODO
    'JMX'    : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = False, rule = r"^$JMX[^]*"), #TODO
    'SSY'    : Grammar('x32',0xe290000000000000, allow_pred = False, reg_store = False, rule = r"^$SSY $i20w24"),
    'SYNC'   : Grammar('x32',0xf0f800000000000f, allow_pred = True, reg_store = False, rule = r"^$SYNC"),
    'CAL'    : Grammar('x32',0xe260000000000040, allow_pred = False, reg_store = False, rule = r"^$CAL $i20w24"),
    'JCAL'   : Grammar('x32',0xe220000000000040, allow_pred = False, reg_store = False, rule = r"^$JCAL $i20w24"),
    'PRET'   : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = False, rule = r"^$PRET[^]*"), #TODO
    'RET'    : Grammar('x32',0xe32000000000000f, allow_pred = True, reg_store = False, rule = r"^$RET"),
    'BRK'    : Grammar('x32',0xe34000000000000f, allow_pred = True, reg_store = False, rule = r"^$BRK"),
    'PBK'    : Grammar('x32',0xe2a0000000000000, allow_pred = False, reg_store = False, rule = r"^$PBK $i20w24"),
    'CONT'   : Grammar('x32',0xe35000000000000f, allow_pred = True, reg_store = False, rule = r"^$CONT"),
    'PCNT'   : Grammar('x32',0xe2b0000000000000, allow_pred = False, reg_store = False, rule = r"^$PCNT $i20w24"),
    'EXIT'   : Grammar('x32',0xe30000000000000f, allow_pred = True, reg_store = False, rule = r"^$EXIT"),
    'PEXIT'  : Grammar('x32',0x0000000000000000, allow_pred = True, reg_store = False, rule = r"^$PEXIT[^]*"), #TODO
    'BPT'    : Grammar('x32',0xe3a00000000000c0, allow_pred = False, reg_store = False, rule = r"^$BPT\.TRAP $i20w24"),

    #Miscellaneous Instructions
    'NOP'    : Grammar('x32', 0x50b0000000000f00, allow_pred = True, reg_store = False, rule = r"^$NOP"),
    'CS2R'   : Grammar('x32', 0x50c8000000000000, allow_pred = True, reg_store = True, rule = r"^$CS2R $r0, $sr"),
    'S2R'    : Grammar('s2r', 0xf0c8000000000000, allow_pred = True, reg_store = True, rule = r"^$S2R $r0, $sr"),
    'B2R'    : Grammar('x32', 0xf0b800010000ff00, allow_pred = True, reg_store = False, rule = r"^$B2R$b2r"),
    'BAR'    : Grammar('gmem',0xf0a8000000000000, allow_pred = True, reg_store = False, rule = r"^$BAR$bar"),
    'DEPBAR' : Grammar('gmem',0xf0f0000000000000, allow_pred = True, reg_store = False, rule = [ r"^$DEPBAR$icmp $dbar, $i20w6", r"^$DEPBAR$dbar2" ]),
    'MEMBAR' : Grammar('x32', 0xef98000000000000, allow_pred = True, reg_store = False, rule = r"^$MEMBAR$mbar"),
    'VOTE'   : Grammar('vote',0x50d8000000000000, allow_pred = True, reg_store = False, rule = r"^$VOTE'vo'e (?:$r0, |(?<nor0>))$p45, $p39"), #TODO: Not sure if reg_store value is correct. Assume False at this point
    'R2B'    : Grammar('x32', 0x0000000000000000, allow_pred = True, reg_store = False, rule = r"^$R2B[^]*"), #TODO


    #Video Instructions... Need to finish
    'VADD'   : Grammar('shft',0x2044000000000000, allow_pred = True, reg_store = True, rule = r"^$VADD$vadd'type''sa'$vaddMode $r0, $r8, $r20, $r39"), #Partial 0x2044000000000000
# TODO
#    'VMAD'   : [
#                  Grammar('x32', 0x5f04000000000000, allow_pred = True, reg_store = True, rule = r"^$VMAD$vmad16 $r0, $r8, $r20, $r39"),
#                  Grammar('shft',0x5f04000000000000, allow_pred = True, reg_store = True, rule = r"^$VMAD$vmad8 $r0, $r8, $r20, $r39"),
#              ],
    'VABSDIFF' : Grammar('shft',0x5427000000000000, allow_pred = True, reg_store = True, rule = r"^$VABSDIFF$vadd'type''sa'$vaddMode $r0, $r8, $r20, $r39"), #Partial 0x2044000000000000
    'VMNMX'    : Grammar('shft',0x3a44000000000000, allow_pred = True, reg_store = True, rule = r"^$VMNMX$vadd'type'$vmnmx'sa'$vaddMode $r0, $r8, $r20, $r39"), #Partial 0x2044000000000000

    'VSET' : Grammar('shft',0x4004000000000000, allow_pred = True, reg_store = True, rule = r"^$VSET$icmp$vadd'type'$vaddMode $r0, $r8, $r20, $r39"), #Partial 0x2044000000000000
}    