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

"""
Class = wrapper to handle lazy logic
expr_ast = expr AST
envr = CURR envr where expr defined
"""
class LazyValue:
    def __init__(self, expr_ast, env_snapshot, interpreter):
        self.expr_ast = expr_ast # expr doesnt enval it
        self.env_snapshot = env_snapshot  # snapshot to capture envr
        self.interpreter = interpreter
        self.cached_value = None
        self.is_evaluated = False
        # print(f"DEBUG: Created LazyValue for {expr_ast}")

    def evaluate(self): # eval expr LAZILY if not alr done + Cache result
        if self.is_evaluated: # if alr eval, return it
            # print(f"DEBUG: LAZYVAL eval -> Using cached value for {self.expr_ast}")
            return self.cached_value
        
        # case for NOT cached valye
        # print(f"DEBUG: LAZYVAL eval -> Evaluating LazyValue: {self.expr_ast}")

        # use snap for eval
        orig_envr = self.interpreter.env
        self.interpreter.env = self.env_snapshot # switch to capture snap
        try:
            self.cached_value = self.interpreter._eval_expr(self.expr_ast)
            self.is_evaluated = True
        finally:
            self.interpreter.env = orig_envr

        # print(f"Cached LazyValue: {self.cached_value.type()}, {self.cached_value.value()}")
        return self.cached_value

