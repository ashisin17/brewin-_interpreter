from intbase import InterpreterBase
from copy import copy

# Enumerated type for our different language data types
class Type:
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    NIL = "nil"


# Represents a value, which has a type and its value
class Value:
    def __init__(self, type, value=None):
        self.t = type
        self.v = value

    def value(self):
        return self.v

    def type(self):
        return self.t
    
    def is_lazy(self):
        return isinstance(self.v, LazyValue)
    
    def evaluate(self):
        if self.is_lazy():
            self.v = self.v.evaluate() # get lazy value
        return self.v


def create_value(val):
    if val == InterpreterBase.TRUE_DEF:
        return Value(Type.BOOL, True)
    elif val == InterpreterBase.FALSE_DEF:
        return Value(Type.BOOL, False)
    elif val == InterpreterBase.NIL_DEF:
        return Value(Type.NIL, None)
    elif isinstance(val, str):
        return Value(Type.STRING, val)
    elif isinstance(val, int):
        return Value(Type.INT, val)
    else:
        raise ValueError("Unknown value type")


def get_printable(val):
    if val.type() == Type.INT:
        return str(val.value())
    if val.type() == Type.STRING:
        return val.value()
    if val.type() == Type.BOOL:
        if val.value() is True:
            return "true"
        return "false"
    return None

class LazyValue:
    def __init__(self, expr_ast, environment, interpreter):
        """
        Class = wrapper to handle lazy logic
        expr_ast = expr AST
        envr = CURR envr where expr defined
        """
        self.expr_ast = expr_ast
        self.environment = environment.snapshot()  # snapshot to capture envr
        self.interpreter = interpreter
        self.cached_value = None
        self.is_evaluated = False

    def evaluate(self): # eval expr LAZILY if not alr done + Cache result
        if not self.is_evaluated: # temp switch to capture env for eval
            previous_env = self.interpreter.env
            self.interpreter.env = self.environment
            try:
                self.cached_value = self.interpreter.__eval_expr(self.expr_ast)
                self.is_evaluated = True
            finally: # restore prev env
                self.interpreter.env = previous_env
        return self.cached_value
