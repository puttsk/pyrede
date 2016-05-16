import json

from pprint import pprint
from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser 

class Sass():
    def __init__(self, sass_file):
        self.header = []
        with open(sass_file) as sass_file:
            for line in sass_file:
                if line[0] == '#':
                    self.header.append(line)
                else:
                    break    
            line = sass_file.readline()
            
            if line != "<CONSTANT_MAPPING>\n":
                raise ValueError("Invalid SASS File")
            
            json_str = ""
            for line in sass_file:
                if line != "</CONSTANT_MAPPING>\n":
                    json_str += line
                else:
                    break
            
            self.constants = json.loads(json_str)
            self.sass_raw = "".join(sass_file.readlines()).strip()
            
def compile(args):
    sass = Sass(args.input_file)
    
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print() 
    
    sass_parser.parse(sass.sass_raw, lexer=sass_lexer)
                
        
        
    
    