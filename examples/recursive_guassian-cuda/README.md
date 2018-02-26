## NOTES

MaxAs has bug when generating binary code for `FCHK.DIVIDE P1, R4, R5` instruction. 
The generated binary code is equivalent to `FCHK.DIVIDE P0, R4, R5` instead, which is incorrect.

### Quick Fix
This problem can be fixed by hard coding in `MaxAsGrammar.pl` by forcing MaxAs to generate binary `0x5c88000000570408` when
encountering `FCHK` instruction. The binary is equivalent to `FCHK.DIVIDE P1, R4, R5`. 

Note that this fix will generate a wrong binary if a program contains FCHK instruction. 

