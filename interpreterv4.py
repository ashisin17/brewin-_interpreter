# document that we won't have a return inside the init/update of a for loop

import copy
from enum import Enum

from brewparse import parse_program
from env_v4 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev4 import Type, Value, LazyValue, create_value, get_printable


class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()
        self.expression_cache = {} # expression cache!

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        self.env = EnvironmentManager()
        self._aux("main", [])

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = {}
            self.func_name_to_ast[func_name][num_params] = func_def

    def __get_func_by_name(self, name, num_params):
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        candidate_funcs = self.func_name_to_ast[name]
        if num_params not in candidate_funcs:
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {name} taking {num_params} params not found",
            )
        return candidate_funcs[num_params]

    def __run_statements(self, statements):
        self.env.push_block()
        # self.expression_cache.clear() # clear cache when entering a new block
        for statement in statements:
            if self.trace_output:
                print(statement)
            status, return_val = self.__run_statement(statement)
            if status == ExecStatus.RETURN:
                self.env.pop_block()
                return (status, return_val)

        self.env.pop_block()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __run_statement(self, statement):
        status = ExecStatus.CONTINUE
        return_val = None
        if statement.elem_type == InterpreterBase.FCALL_NODE:
            self._call_func(statement)
        elif statement.elem_type == "=":
            self.__assign(statement)
        elif statement.elem_type == InterpreterBase.VAR_DEF_NODE:
            self.__var_def(statement)
        elif statement.elem_type == InterpreterBase.RETURN_NODE:
            status, return_val = self.__do_return(statement)
        elif statement.elem_type == Interpreter.IF_NODE:
            status, return_val = self.__do_if(statement)
        elif statement.elem_type == Interpreter.FOR_NODE:
            status, return_val = self.__do_for(statement)

        return (status, return_val)
    
    def _call_func(self, call_node):
        func_name = call_node.get("name")
        actual_args = call_node.get("args")
        return self._aux(func_name, actual_args)

    def _aux(self, func_name, actual_args):
        if func_name == "print":
            return self.__call_print(actual_args)
        if func_name == "inputi" or func_name == "inputs":
            return self.__call_input(func_name, actual_args)

        func_ast = self.__get_func_by_name(func_name, len(actual_args))
        formal_args = func_ast.get("args")
        if len(actual_args) != len(formal_args):
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
            )

        # first evaluate all of the actual parameters and associate them with the formal parameter names
        args = {}
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            # result = copy.copy(self._eval_expr(actual_ast))
            lazy_arg = LazyValue(actual_ast, self.env.snapshot(), self) # update args to have lazy eval
            arg_name = formal_ast.get("name")
            args[arg_name] = Value(Type.NIL, lazy_arg)

        # then create the new activation record 
        self.env.push_func()
        # and add the formal arguments to the activation record
        for arg_name, value in args.items():
          self.env.create(arg_name, value)
        _, return_val = self.__run_statements(func_ast.get("statements"))
        self.env.pop_func()

        # validate return type for lazy value!
        if return_val.type() == Type.NIL and isinstance(return_val.value(), LazyValue):
            # print(f"DEBUG: Function {func_name} returned a LazyValue. Evaluating...")
            return_val = return_val.value().evaluate()
        return return_val

    def __call_print(self, args):
        output = ""
        for arg in args:
            result = self._eval_expr(arg)  # result is a Value object
            # eager eval in specific #2: print as built iin
            if isinstance(result.value(), LazyValue):
                result = result.value().evaluate()
            output += get_printable(result)

        super().output(output)
        return Interpreter.NIL_VALUE

    def __call_input(self, name, args): # eager eval #2: built in types
        if args is not None and len(args) == 1:
            result = self._eval_expr(args[0])
            if isinstance(result.value(), LazyValue):
                result = result.value().evaluate()

            if result.type() != Type.STRING:
                super().error(ErrorType.TYPE_ERROR, "input function argument must be a string")

        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )

        super().output(get_printable(result))
        inp = super().get_input()

        if name == "inputi":
            return Value(Type.INT, int(inp))
        if name == "inputs":
            return Value(Type.STRING, inp)

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        expr_ast = assign_ast.get("expression") # lazy eval -> dont eval yet, get expr

        # evaluated_value = self._eval_expr(expr_ast)
        # if evaluated_value.is_lazy():
        #     # print(f"DEBUG: Evaluating lazy value for assignment to {var_name}")
        #     evaluated_value = evaluated_value.evaluate()

        # # Invalidate cache entries involving this variable
        # # print(f"DEBUG:ASSIGN! Clearing cache entries for variable {var_name}")
        # keys_invalidate = [k for k in self.expression_cache if f"var: name: {var_name}" in str(k)]
        # for key in keys_invalidate:
        #     # print(f"DEBUG: Invalidating cache for {key}")
        #     del self.expression_cache[key]

        lazy_value = LazyValue(expr_ast, self.env.snapshot(), self)
        value_obj = Value(Type.NIL, lazy_value)

        if not self.env.set(var_name, value_obj):
            super().error(
                ErrorType.NAME_ERROR, f"Undefined variable {var_name} in assignment"
            )
    
    def __var_def(self, var_ast):
        var_name = var_ast.get("name")
        if not self.env.create(var_name, Interpreter.NIL_VALUE):
            super().error(
                ErrorType.NAME_ERROR, f"Duplicate definition for variable {var_name}"
            )

    def _eval_expr(self, expr_ast):
        # generate unique KEY for each expression
        if expr_ast.elem_type == InterpreterBase.FCALL_NODE:
            func_name = expr_ast.get("name")
            args = tuple(str(arg) for arg in expr_ast.get("args"))  # use args as cache key
            expr_key = ("fcall", func_name, args)
        else:
            expr_key = str(expr_ast) # for expr, use string expr
        
        # print(f"DEBUG: Current cache keys: {list(self.expression_cache.keys())}")
        # print(f"DEBUG: Checking for expr_key: {expr_key}")

        # if value or expr already cached, return that!
        if expr_key in self.expression_cache: 
            # print(f"DEBUG: RETURN cached val for eval expr -> Using cached LazyValue for {expr_key}")
            cached_value = self.expression_cache[expr_key]

            if cached_value.is_lazy(): # lazyval inside cached val is EVAL!
                # print(f"DEBUG: Evaluating cached LazyValue for {expr_key}")
                cached_value = cached_value.evaluate()
                self.expression_cache[expr_key] = cached_value
            
            if cached_value.type() == Type.NIL:
                # print(f"DEBUG: Cached value is NIL for {expr_key}. Reevaluating...")
                cached_value = cached_value.evaluate()
                self.expression_cache[expr_key] = cached_value  # Update the cache
                if cached_value.type() == Type.NIL:  # Still NIL after reevaluation
                    super().error(
                        ErrorType.TYPE_ERROR,
                        f"Cached LazyValue resulted in NIL after reevaluation: {expr_key}"
                    )

            return cached_value
        
        if expr_ast.elem_type == InterpreterBase.FCALL_NODE: # eager eval
            # print(f"DEBUG: INSIDE FCALL for eval expr -> Creating new LazyValue for {expr_key}")
            lazy_val = LazyValue(expr_ast, self.env, self)

            evaluated_value = Value(Type.NIL, lazy_val) # imm eval and cache result
            self.expression_cache[expr_key] = evaluated_value
            return evaluated_value

        # simple literals nil, int, string, bool -> NOT affected
        if expr_ast.elem_type == InterpreterBase.NIL_NODE: 
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_NODE:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_NODE:
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_NODE:
            return Value(Type.BOOL, expr_ast.get("val"))
        
        # lazy eval affected!
        if expr_ast.elem_type == InterpreterBase.VAR_NODE:
            var_name = expr_ast.get("name")
            val = self.env.get(var_name)
            if val is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            
            # handle lazy eval
            if isinstance(val.value(), LazyValue):
                return val.value().evaluate()
            return val
        
        if expr_ast.elem_type in Interpreter.BIN_OPS: # update binary-> eval op to handle lazy                
            return self.__eval_op(expr_ast)
        
        if expr_ast.elem_type == Interpreter.NEG_NODE: #lazy eval for not and neg?
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x)
        
        if expr_ast.elem_type == Interpreter.NOT_NODE:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)

    def __eval_op(self, arith_ast):
        left_value_obj = self._eval_expr(arith_ast.get("op1"))
        right_value_obj = self._eval_expr(arith_ast.get("op2"))

        # print(f"DEBUG: EVAL OP for: {arith_ast.elem_type}")
        # print("BEFORE lazy eval in eval op")
        # print(f"DEBUG: Left operand: {left_value_obj.type()}, {left_value_obj.value()}")
        # print(f"DEBUG: Right operand: {right_value_obj.type()}, {right_value_obj.value()}")
        # handle lazy obj here
        if isinstance(left_value_obj.value(), LazyValue): # handle nested valueS!
            left_value_obj = left_value_obj.value().evaluate()
        if isinstance(right_value_obj.value(), LazyValue):
            right_value_obj = right_value_obj.value().evaluate()

        if left_value_obj.type() == Type.NIL or right_value_obj.type() == Type.NIL:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Invalid operand for {arith_ast.elem_type} operation: Left({left_value_obj.type()}), Right({right_value_obj.type()})"
            )

        # print("AFTER lazy eval in eval op")
        # print(f"DEBUG: Left operand: {left_value_obj.type()}, {left_value_obj.value()}")
        # print(f"DEBUG: Right operand: {right_value_obj.type()}, {right_value_obj.value()}")
            
        # check for compatible types
        if not self.__compatible_types(
            arith_ast.elem_type, left_value_obj, right_value_obj
        ):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {arith_ast.elem_type} operation",
            )
        if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {arith_ast.elem_type} for type {left_value_obj.type()}",
            )
        # perform the op
        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
        result = f(left_value_obj, right_value_obj)

        # cache result for future reuse
        return Value(result.type(), result.value())

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f):
        value_obj = self._eval_expr(arith_ast.get("op1"))

        if isinstance(value_obj.value(), LazyValue): # if its lazy eval it
            value_obj = value_obj.value().evaluate()
        
        # print(f"Unary operation operand type: {value_obj.type()}, value: {value_obj.value()}")

        if value_obj.type() != t:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible type for {arith_ast.elem_type} operation",
            )
        return Value(t, f(value_obj.value()))

    def __setup_ops(self):
        self.op_to_lambda = {}
        # set up operations on integers
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), x.value() - y.value()
        )
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), x.value() * y.value()
        )
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            x.type(), x.value() // y.value()
        )
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )
        #  set up operations on strings
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.STRING]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.STRING]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        #  set up operations on bools
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            x.type(), x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            x.type(), x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        result = self._eval_expr(cond_ast)

        # eager eval: conditional #1
        if result.is_lazy():
            result = result.evaluate()

        if result.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR,
                "Incompatible type for if condition",
            )
        if result.value():
            statements = if_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            return (status, return_val)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                status, return_val = self.__run_statements(else_statements)
                return (status, return_val)

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_for(self, for_ast):
        init_ast = for_ast.get("init") 
        cond_ast = for_ast.get("condition")
        update_ast = for_ast.get("update") 

        self.__run_statement(init_ast)  # initialize counter variable

        # run_for = Interpreter.TRUE_VALUE
        while True:
            run_for = self._eval_expr(cond_ast)  # check for-loop condition ONCE

            #eager eval: conditional #1 FOR
            if run_for.is_lazy():
                run_for = run_for.evaluate()

            if run_for.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for for condition",
                )
            
            if not run_for.value(): # exit loop when condition is false
                break

            if run_for.value():
                statements = for_ast.get("statements")
                status, return_val = self.__run_statements(statements)

                if status == ExecStatus.RETURN:
                    return status, return_val
                self.__run_statement(update_ast)  # update counter variable

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return (ExecStatus.RETURN, Interpreter.NIL_VALUE)
        
        value_obj = copy.copy(self._eval_expr(expr_ast))
        # lazy eval for return
        if isinstance(value_obj.value(), LazyValue):
            value_obj = value_obj.value().evaluate()
        return (ExecStatus.RETURN, value_obj)
    
    #TODO: do raise stuff once the node is built up
    """
    def __do_raise(self, raise_ast):
    exception_expr = raise_ast.get("exception_type")
    exception_value = self._eval_expr(exception_expr)
    # Eagerly evaluate LazyValues
    if isinstance(exception_value.value(), LazyValue):
        exception_value = exception_value.value().evaluate()
    if exception_value.type() != Type.STRING:
        super().error(ErrorType.TYPE_ERROR, "Exception must be a string")
    raise BrewinException(exception_value.value())
    """
