from collections import namedtuple

Grammar = namedtuple('Grammar', ['type', 'code', 'rule']) 

SASS_GRAMMARS = {    
    #Floating Point Instructions
    'FADD'     : [ Grammar('x32', 0x5c58000000000000, r"^$pred?FADD'f'z$rnd'sa' $r0, $r8, $fcr20;") ],
    'FADD32I'  : [ Grammar('x32', 0x0800000000000000, r"^$pred?FADD32I'f'z $r0, $r8, $f20w32;") ],
    'FCHK'     : [ Grammar('x32', 0x5c88000000000000, r"^$pred?FCHK\.DIVIDE $p0, $r8, $r20;") ], #Partial?
    'FCMP'     : [ Grammar('cmp', 0x5ba0000000000000, r"^$pred?FCMP$fcmp'f'z $r0, $r8, $fcr20, $r39;") ],
    'FFMA'     : [
                  Grammar('x32', 0x5980000000000000, r"^$pred?FFMA'f'z$rnd'sa' $r0, $r8, $fcr20, $r39;"),
                  Grammar('x32', 0x5980000000000000, r"^$pred?FFMA'f'z$rnd'sa' $r0, $r8, $r39s20, $c20s39;"),
                ],
    'FMNMX'    : [ Grammar('shft',0x5c60000000000000, r"^$pred?FMNMX'f'z $r0, $r8, $fcr20, $p39;") ],
    'FMUL'     : [ Grammar('x32', 0x5c68000000000000, r"^$pred?FMUL'f'z$rnd'sa' $r0, $r8, $fcr20;") ],
    'FMUL32I'  : [ Grammar('x32', 0x1e00000000000000, r"^$pred?FMUL32I'f'z $r0, $r8, $f20w32;") ],
    'FSET'     : [ Grammar('shft',0x5800000000000000, r"^$pred?FSET$fcmp'f'z$bool $r0, $r8, $fcr20, $p39;") ],
    'FSETP'    : [ Grammar('cmp', 0x5bb0000000000000, r"^$pred?FSETP$fcmp'f'z$bool $p3, $p0, $r8, $fcr20, $p39;") ],
    'MUFU'     : [ Grammar('qtr', 0x5080000000000000, r"^$pred?MUFU$func $r0, $r8;") ],
    'RRO'      : [ Grammar('rro', 0x5c90000000000000, r"^$pred?RRO$rro $r0, $r20;") ],
    'DADD'     : [ Grammar('x64', 0x5c70000000000000, r"^$pred?DADD$rnd $r0, $r8, $dr20;") ],
    'DFMA'     : [ Grammar('x64', 0x5b70000000000000, r"^$pred?DFMA$rnd $r0, $r8, $dr20, $r39;") ],
    'DMNMX'    : [ Grammar('cmp', 0x5c50000000000000, r"^$pred?DMNMX $r0, $r8, $dr20, $p39;") ],
    'DMUL'     : [ Grammar('x64', 0x5c80000000000000, r"^$pred?DMUL$rnd $r0, $r8, $dr20;") ],
    'DSET'     : [ Grammar('cmp', 0x5900000000000000, r"^$pred?DSET$fcmp$bool $r0, $r8, $dr20, $p39;") ],
    'DSETP'    : [ Grammar('cmp', 0x5b80000000000000, r"^$pred?DSETP$fcmp$bool $p3, $p0, $r8, $dr20, $p39;") ],
    'FSWZADD'  : [ Grammar('x32', 0x0000000000000000, r"^$pred?FSWZADD[^;]*;") ], #TODO

    'HADD2'     : [ Grammar('x32', 0x5d10000000000000, r"^$pred?HADD2'f'z $r0, $r8, $r20;") ],
    'HMUL2'     : [ Grammar('x32', 0x5d08000000000000, r"^$pred?HMUL2'f'z $r0, $r8, $r20;") ],
    'HFMA2'     : [ Grammar('x32', 0x5d00000000000000, r"^$pred?HFMA2'f'z $r0, $r8, $r20, $r39;") ],
    'HSETP2'    : [ Grammar('cmp', 0x5d20000000000000, r"^$pred?HSETP2$fcmp$bool $p3, $p0, $r8, $fcr20, $p39;") ], #Partial

    #Integer Instructions
    'BFE'       : [ Grammar('shft', 0x5c01000000000000, r"^$pred?BFE$u32 $r0, $r8, $icr20;") ],
    'BFI'       : [ Grammar('shft', 0x5bf0000000000000, r"^$pred?BFI $r0, $r8, $ir20, $cr39;") ],
    'FLO'       : [ Grammar('s2r',  0x5c30000000000000, r"^$pred?FLO\.U32 $r0, $icr20;") ],
    'IADD'      : [ Grammar('x32',  0x5c10000000000000, r"^$pred?IADD'sa'$X $r0cc, $r8, $icr20;") ],
    'IADD32I'   : [ Grammar('x32',  0x1c00000000000000, r"^$pred?IADD32I$X $r0cc, $r8, $i20w32;") ],
    'IADD3'     : [ Grammar('x32',  0x5cc0000000000000, r"^$pred?IADD3$add3 $r0cc, $r8, $icr20, $r39;") ],
    'ICMP'      : [ Grammar('cmp',  0x5b41000000000000, r"^$pred?ICMP$icmp$u32 $r0, $r8, $icr20, $r39;") ],
    'IMNMX'     : [ Grammar('shft', 0x5c21000000000000, r"^$pred?IMNMX$u32$hilo $r0cc, $r8, $icr20, $p39;") ],
    'ISET'      : [ Grammar('shft', 0x5b51000000000000, r"^$pred?ISET$icmp$u32$X$bool $r0, $r8, $icr20, $p39;") ],
    'ISETP'     : [ Grammar('cmp',  0x5b61000000000000, r"^$pred?ISETP$icmp$u32$X$bool $p3, $p0, $r8, $icr20, $p39;") ],
    'ISCADD'    : [ Grammar('shft', 0x5c18000000000000, r"^$pred?ISCADD $r0, $r8, $icr20, $i39w8;") ],
    'ISCADD32I' : [ Grammar('shft', 0x1400000000000000, r"^$pred?ISCADD32I $r0, $r8, $i20w32, $i53w5;") ],
    'LEA'       : [
                   Grammar('cmp',  0x5bd0000000000000, r"^$pred?LEA $p48, $r0cc, $r8, $icr20;"),
                   Grammar('shft', 0x5bd7000000000000, r"^$pred?LEA $r0cc, $r8, $icr20, $i39w8;"),
                   Grammar('shft', 0x5bdf004000000000, r"^$pred?LEA\.HI$X $r0cc, $r8, $r20, $r39, $i28w8;"),
                   Grammar('shft', 0x0a07000000000000, r"^$pred?LEA\.HI$X $r0cc, $r8, $c20, $r39, $i51w5;"),
                 ],
    'LOP'       : [ Grammar('x32',  0x5c40000000000000, r"^$pred?LOP$bool$lopz $r0, $r8, (?<INV>~)?$icr20(?<INV>\.INV)?;") ],
    'LOP32I'    : [ Grammar('x32',  0x0400000000000000, r"^$pred?LOP32I$bool $r0, $r8, $i20w32;") ],
    'LOP3'      : [
                   Grammar('x32',  0x5be7000000000000, r"^$pred?LOP3\.LUT $r0, $r8, $r20, $r39, $i28w8;"),
                   Grammar('x32',  0x3c00000000000000, r"^$pred?LOP3\.LUT $r0, $r8, $i20, $r39, $i48w8;"),
                 ],
    'POPC'      : [ Grammar('s2r',  0x5c08000000000000, r"^$pred?POPC $r0, $r20;") ],
    'SHF'       : [
                   Grammar('shft', 0x5bf8000000000000, r"^$pred?SHF\.L$shf $r0, $r8, $ir20, $r39;"),
                   Grammar('shft', 0x5cf8000000000000, r"^$pred?SHF\.R$shf $r0, $r8, $ir20, $r39;"),
                 ],
    'SHL'       : [ Grammar('shft', 0x5c48000000000000, r"^$pred?SHL(?<W>\.W)? $r0, $r8, $icr20;") ],
    'SHR'       : [ Grammar('shft', 0x5c29000000000000, r"^$pred?SHR$u32 $r0, $r8, $icr20;") ],
    'XMAD'      : [
                   Grammar('x32',  0x5b00000000000000, r"^$pred?XMAD$xmad $r0cc, $r8, $ir20, $r39;"),
                   Grammar('x32',  0x5900000000000000, r"^$pred?XMAD$xmad $r0cc, $r8, $r39s20, $c20s39;"),
                   Grammar('x32',  0x5e00000000000000, r"^$pred?XMAD$xmadc $r0cc, $r8, $c20x, $r39;"),
                 ],
    # XMAD replaces these
    'IMAD'      : [ Grammar('x32',  0x0000000000000000, r"^$pred?IMAD[^;]*;") ], #TODO
    'IMADSP'    : [ Grammar('x32',  0x0000000000000000, r"^$pred?IMADSP[^;]*;") ], #TODO
    'IMUL'      : [ Grammar('x32',  0x0000000000000000, r"^$pred?IMUL[^;]*;") ], #TODO

    #Conversion Instructions
    'F2F' : [ Grammar('qtr', 0x5ca8000000000000, r"^$pred?F2F'f'z$x2x$rnd$round'sa' $r0, $cr20;") ],
    'F2I' : [ Grammar('qtr', 0x5cb0000000000000, r"^$pred?F2I'f'z$x2x$round $r0, $cr20;") ],
    'I2F' : [ Grammar('qtr', 0x5cb8000000000000, r"^$pred?I2F$x2x$rnd $r0, $cr20;") ],
    'I2I' : [ Grammar('qtr', 0x5ce0000000000000, r"^$pred?I2I$x2x'sa' $r0, $cr20;") ],

    #Movement Instructions
    'MOV'    : [ Grammar('x32', 0x5c98078000000000, r"^$pred?MOV $r0, $icr20;") ],
    'MOV32I' : [ Grammar('x32', 0x010000000000f000, r"^$pred?MOV32I $r0, (?:$i20w32|$f20w32);") ],
    'PRMT'   : [ Grammar('x32', 0x5bc0000000000000, r"^$pred?PRMT'prm' $r0, $r8, $icr20, $cr39;") ],
    'SEL'    : [ Grammar('x32', 0x5ca0000000000000, r"^$pred?SEL $r0, $r8, $icr20, $p39;") ],
    'SHFL'   : [ Grammar('smem',0xef10000000000000, r"^$pred?SHFL$shfl $p48, $r0, $r8, (?:$i20w8|$r20), (?:$i34w13|$r39);") ],

    #Predicate/CC Instructions
    'PSET'   : [ Grammar('cmp', 0x5088000000000000, r"^$pred?PSET$bool2$bool $r0, $p12, $p29, $p39;") ],
    'PSETP'  : [ Grammar('cmp', 0x5090000000000000, r"^$pred?PSETP$bool2$bool $p3, $p0, $p12, $p29, $p39;") ],
    'CSET'   : [ Grammar('x32', 0x0000000000000000, r"^$pred?CSET[^;]*;") ], #TODO
    'CSETP'  : [ Grammar('x32', 0x0000000000000000, r"^$pred?CSETP[^;]*;") ], #TODO
    'P2R'    : [ Grammar('x32', 0x38e8000000000000, r"^$pred?P2R $r0, PR, $r8, $i20w7;") ],
    'R2P'    : [ Grammar('shft',0x38f0000000000000, r"^$pred?R2P PR, $r8, $i20w7;") ],

    #Texture Instructions
    # Handle the commonly used 1D texture functions.. but save the others for later
    'TLD'    : [ Grammar('gmem',0xdd38000000000000, r"^$pred?TLD\.B\.LZ\.$tld $r0, $r8, $r20, $hex, \dD, $i31w4;") ], #Partial
    'TLDS'   : [ Grammar('gmem',0xda0000000ff00000, r"^$pred?TLDS\.LZ\.$tld $r28, $r0, $r8, $i36w20, \dD, $chnls;",) ], #Partial
    'TEX'    : [ Grammar('gmem',0x0000000000000000, r"^$pred?TEX[^;]*;") ], #TODO
    'TLD4'   : [ Grammar('gmem',0x0000000000000000, r"^$pred?TLD4[^;]*;") ], #TODO
    'TXQ'    : [ Grammar('gmem',0x0000000000000000, r"^$pred?TXQ[^;]*;") ], #TODO
    'TEXS'   : [ Grammar('gmem',0x0000000000000000, r"^$pred?TEXS[^;]*;") ], #TODO
    'TLD4S'  : [ Grammar('gmem',0x0000000000000000, r"^$pred?TLD4S[^;]*;") ], #TODO

    #Compute Load/Store Instructions
    'LD'     : [ Grammar('gmem',0x8000000000000000, r"^$pred?LD$memCache$mem'type' $r0, $addr, $p58;") ],
    'ST'     : [ Grammar('gmem',0xa000000000000000, r"^$pred?ST$memCache$mem'type' $addr, $r0, $p58;") ],
    'LDG'    : [ Grammar('gmem',0xeed0000000000000, r"^$pred?LDG$memCache$mem'type' $r0, $addr;") ],
    'STG'    : [ Grammar('gmem',0xeed8000000000000, r"^$pred?STG$memCache$mem'type' $addr, $r0;") ],
    'LDS'    : [ Grammar('smem',0xef48000000000000, r"^$pred?LDS$memCache$mem'type' $r0, $addr;") ],
    'STS'    : [ Grammar('smem',0xef58000000000000, r"^$pred?STS$memCache$mem'type' $addr, $r0;") ],
    'LDL'    : [ Grammar('gmem',0xef40000000000000, r"^$pred?LDL$memCache$mem'type' $r0, $addr;") ],
    'STL'    : [ Grammar('gmem',0xef50000000000000, r"^$pred?STL$memCache$mem'type' $addr, $r0;") ],
    'LDC'    : [ Grammar('gmem',0xef90000000000000, r"^$pred?LDC$memCache$mem'type' $r0, $ldc;") ],
    # Note for ATOM(S).CAS operations the last register needs to be in sequence with the second to last (as it's not en'code'd).
    'ATOM'   : [ Grammar('gmem',0xed00000000000000, r"^$pred?ATOM'a'om $r0, $addr2, $r20(?:, $r39a)?;") ],
    'ATOMS'  : [ Grammar('smem',0xec00000000000000, r"^$pred?ATOMS'a'om $r0, $addr2, $r20(?:, $r39a)?;") ],
    'RED'    : [ Grammar('gmem',0xebf8000000000000, r"^$pred?RED'a'om $addr2, $r0;") ],
    'CCTL'   : [ Grammar('x32', 0x5c88000000000000, r"^$pred?CCTL[^;]*;") ], #TODO
    'CCTLL'  : [ Grammar('x32', 0x5c88000000000000, r"^$pred?CCTLL[^;]*;") ], #TODO
    'CCTLT'  : [ Grammar('x32', 0x5c88000000000000, r"^$pred?CCTLT[^;]*;") ], #TODO

    #Surface Memory Instructions (haven't gotten to these yet..)
    'SUATOM' : [ Grammar('x32',0x0000000000000000, r"^$pred?SUATOM[^;]*;") ], #TODO
    'SULD'   : [ Grammar('x32',0x0000000000000000, r"^$pred?SULD[^;]*;") ], #TODO
    'SURED'  : [ Grammar('x32',0x0000000000000000, r"^$pred?SURED[^;]*;") ], #TODO
    'SUST'   : [ Grammar('x32',0x0000000000000000, r"^$pred?SUST[^;]*;") ], #TODO

    #Control Instructions
    'BRA'    : [
                Grammar('x32',0xe24000000000000f, r"^$pred?BRA(?<U>\.U)? $i20w24;"),
                Grammar('x32',0xe240000000000002, r"^$pred?BRA(?<U>\.U)? CC\.EQ, $i20w24;"),
              ],
    'BRX'    : [ Grammar('x32',0x0000000000000000, r"^$pred?BRX[^;]*;") ], #TODO
    'JMP'    : [ Grammar('x32',0x0000000000000000, r"^$pred?JMP[^;]*;") ], #TODO
    'JMX'    : [ Grammar('x32',0x0000000000000000, r"^$pred?JMX[^;]*;") ], #TODO
    'SSY'    : [ Grammar('x32',0xe290000000000000, r"^$noPred?SSY $i20w24;") ],
    'SYNC'   : [ Grammar('x32',0xf0f800000000000f, r"^$pred?SYNC;") ],
    'CAL'    : [ Grammar('x32',0xe260000000000040, r"^$noPred?CAL $i20w24;") ],
    'JCAL'   : [ Grammar('x32',0xe220000000000040, r"^$noPred?JCAL $i20w24;") ],
    'PRET'   : [ Grammar('x32',0x0000000000000000, r"^$pred?PRET[^;]*;") ], #TODO
    'RET'    : [ Grammar('x32',0xe32000000000000f, r"^$pred?RET;") ],
    'BRK'    : [ Grammar('x32',0xe34000000000000f, r"^$pred?BRK;") ],
    'PBK'    : [ Grammar('x32',0xe2a0000000000000, r"^$noPred?PBK $i20w24;") ],
    'CONT'   : [ Grammar('x32',0xe35000000000000f, r"^$pred?CONT;") ],
    'PCNT'   : [ Grammar('x32',0xe2b0000000000000, r"^$noPred?PCNT $i20w24;") ],
    'EXIT'   : [ Grammar('x32',0xe30000000000000f, r"^$pred?EXIT;") ],
    'PEXIT'  : [ Grammar('x32',0x0000000000000000, r"^$pred?PEXIT[^;]*;") ], #TODO
    'BPT'    : [ Grammar('x32',0xe3a00000000000c0, r"^$noPred?BPT\.TRAP $i20w24;") ],

    #Miscellaneous Instructions
    'NOP'    : [ Grammar('x32', 0x50b0000000000f00, r"^$pred?NOP;") ],
    'CS2R'   : [ Grammar('x32', 0x50c8000000000000, r"^$pred?CS2R $r0, $sr;") ],
    'S2R'    : [ Grammar('s2r', 0xf0c8000000000000, r"^$pred?S2R $r0, $sr;") ],
    'B2R'    : [ Grammar('x32', 0xf0b800010000ff00, r"^$pred?B2R$b2r;") ],
    'BAR'    : [ Grammar('gmem',0xf0a8000000000000, r"^$pred?BAR$bar;") ],
    'DEPBAR' : [
                Grammar('gmem',0xf0f0000000000000, r"^$pred?DEPBAR$icmp $dbar, $i20w6;"),
                Grammar('gmem',0xf0f0000000000000, r"^$pred?DEPBAR$dbar2;"),
              ],
    'MEMBAR' : [ Grammar('x32', 0xef98000000000000, r"^$pred?MEMBAR$mbar;") ],
    'VOTE'   : [ Grammar('vote',0x50d8000000000000, r"^$pred?VOTE'vo'e (?:$r0, |(?<nor0>))$p45, $p39;") ],
    'R2B'    : [ Grammar('x32', 0x0000000000000000, r"^$pred?R2B[^;]*;") ], #TODO

    #Video Instructions... Need to finish
    'VADD'   : [   Grammar('shft',0x2044000000000000, r"^$pred?VADD$vadd'type''sa'$vaddMode $r0, $r8, $r20, $r39;") ], #Partial 0x2044000000000000
    'VMAD'   : [
                  Grammar('x32', 0x5f04000000000000, r"^$pred?VMAD$vmad16 $r0, $r8, $r20, $r39;"),
                  Grammar('shft',0x5f04000000000000, r"^$pred?VMAD$vmad8 $r0, $r8, $r20, $r39;"),
              ],
    'VABSDIFF' : [ Grammar('shft',0x5427000000000000, r"^$pred?VABSDIFF$vadd'type''sa'$vaddMode $r0, $r8, $r20, $r39;") ], #Partial 0x2044000000000000
    'VMNMX'    : [ Grammar('shft',0x3a44000000000000, r"^$pred?VMNMX$vadd'type'$vmnmx'sa'$vaddMode $r0, $r8, $r20, $r39;") ], #Partial 0x2044000000000000

    'VSET' : [ Grammar('shft',0x4004000000000000, r"^$pred?VSET$icmp$vadd'type'$vaddMode $r0, $r8, $r20, $r39;") ], #Partial 0x2044000000000000
}    