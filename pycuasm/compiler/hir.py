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
        return "%d:%d:%d:%d:%d" % (
            self.wait_barrier,
            self.read_barrier,
            self.write_barrier,
            self.yield_hint,
            self.stall
        )
    
        
        
