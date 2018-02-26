Sample: Recursive Gaussian Filter
Minimum spec: SM 1.1

This sample implements a Gaussian blur using Deriche's recursive method. The advantage of this method is that the execution time is independent of the filter width.

Key concepts:

Notes: 
MaxAs generates wrong code when generating binary code for FCHK.DIVIDE P1, R4, R5 instruction. 
The generated binary code is equivalent to FCHK.DIVIDE P0, R4, R5 instead, which is incorrect.
This problem is fixed by hard coding in MaxAsGrammar.pl. MaxAs will generate binary 0x5c88000000570408 when 
encountering FCHK opcode, which is equivalent to FCHK.DIVIDE P1, R4, R5. If any program contains 
FCHK instruction, this code in MaxAsGrammar.pl must be commented out. 
