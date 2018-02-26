class BarrierTracker(object):
    def __init__(self):
        # Unset all barrier to
        # '-' : unset
        # 'r' : read barrier
        # 'w' : write barrier 
        self.barriers = ['-', '-', '-', '-', '-', '-']
        self.__last_read_flag = 1
        self.__last_write_flag = 1

    def __repr__(self):
        return repr(self.barriers)

    def reset(self):
        self.barriers = ['-', '-', '-', '-', '-', '-']
        
    def is_available(self, flag):
        if self.barriers[flag-1] != '-':
            return False
        else:
            return True
        
    def update_flags(self, flags):
        if flags.wait_barrier != 0:
            for i in range(6):
                if flags.wait_barrier & 2**i != 0:
                    self.barriers[i] = '-'
    
        if flags.read_barrier != 0:
            self.barriers[flags.read_barrier-1] = 'r'
            self.__last_read_flag = flags.read_barrier
            
        if flags.write_barrier != 0:
            self.barriers[flags.write_barrier-1] = 'w'
            self.__last_write_flag = flags.write_barrier
    
    def get_available_flags(self, mode):
        if '-' in self.barriers:
            free_barrier = self.barriers.index('-')
        else:
            if mode == 'r':
                free_barrier = self.__last_read_flag - 1
            else:
                free_barrier = self.__last_write_flag - 1
                
        self.barriers[free_barrier] = mode
        
        return free_barrier+1