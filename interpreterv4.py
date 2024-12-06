
# document that we won't have a return inside the init/update of a for loop

import copy
from enum import Enum

from brewparse import parse_program
from env_v4 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev4 import Type, Value, create_value, get_printable

"""
WRAPPER class to save the lazy value -> putting it in the same file as the interpreter 
so no issues with calling it!
"""
class LazyValue: 
    def __init__(self, expr_ast, env_snapshot): # no need to pass in interpreter
        self.expr_ast = expr_ast # closure implemenaation -> save the expresison ast
        self.cached_value = None
        self.evaluated = False
        self.env_snapshot = env_snapshot  # envr snapshot

    def evaluate(self, interpreter):
        if self.evaluated:
            return self.cached_value
        # got direct val obj, so return as is
        if isinstance(self.expr_ast, Value):  
            self.cached_value = self.expr_ast
        else:
            try:
                original_env = interpreter.env.environment
                interpreter.env.environment = self.env_snapshot
                self.cached_value = interpreter._eval_expr(self.expr_ast)  # eval func directly called
                self.evaluated = True
            except Exception as e: # reraise xception to propagate!
                interpreter.env.environment = original_env
                raise e
            finally: # ensures that the orig env restores NO MATTERWHAT
                interpreter.env.environment = original_env  # restore environment
        return self.cached_value

# EXCEPTION HANDLING CLASS
class InterpreterException(Exception):
   def __init__(self, exception_name):
       self.exception_name = exception_name

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

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        self.env = EnvironmentManager()
        self.__call_func_aux("main", [])

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
        for statement in statements:
            if self.trace_output:
                print(statement)
            status, return_val = self.__run_statement(statement)
            if status == ExecStatus.RETURN:
                self.env.pop_block()
                return (status, return_val)

        # only pop if it has the blocks!
        if self.env.has_blocks():  
            self.env.pop_block()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __run_statement(self, statement):
        status = ExecStatus.CONTINUE
        return_val = None
        if statement.elem_type == InterpreterBase.FCALL_NODE:
            self.__call_func(statement)
        elif statement.elem_type == "raise": # add the raise!
            self.__handle_raise(statement)
        elif statement.elem_type == "try": # add try catch block
            return self.__handle_try(statement)
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
    
    # add raise functionality!
    def __handle_raise(self, statement):
        expr_ast = statement.get("exception_type")
        if expr_ast is None: #expr ast must exist + be valid! -> add check: or not isinstance(expr_ast, dict)
            super().error(ErrorType.TYPE_ERROR, "Invalid expression for raise")
        
        # EAGER eval for raise
        except_val = self._eval_expr(expr_ast)
        except_val = self.eval_asap(except_val)  # make sure lazy val is for sure resolved
        
        if not isinstance(except_val.value(), str): # exception type must be string!
            super().error(ErrorType.TYPE_ERROR, "Exception must be a string")
        
        raise InterpreterException(except_val.value())
    
    # add try functionality!
    def __handle_try(self, statement):
        try_b = statement.get("statements")
        catch_bs = statement.get("catchers")  # multiple catch blocks

        try:
            # new scope for try to reset!
            self.env.push_block()  
            status, return_val = self.__run_statements(try_b)
            self.env.pop_block()
            return status, return_val
        except Exception as e:
            # try block HAS to be cleared after!
            if self.env.has_blocks():
                self.env.pop_block()

            exception_type = e.exception_name
            for catch_block in catch_bs:
                catch_type = catch_block.get("exception_type")  # get excep type
                if catch_type.strip("\"") == exception_type:  # MATCH string literal by removing quotes
                    self.env.push_block()  # NEW SCOPE for catch block
                    status, return_val = self.__run_statements(catch_block.get("statements"))
                    self.env.pop_block()
                    return status, return_val

            raise e # prop exception if no match

    def __call_func(self, call_node):
        func_name = call_node.get("name")
        actual_args = call_node.get("args")
        return self.__call_func_aux(func_name, actual_args)

    def __call_func_aux(self, func_name, actual_args):
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
            result = copy.copy(self._eval_expr(actual_ast))
            arg_name = formal_ast.get("name")
            args[arg_name] = result
            # Should we do lazy eval for args?
            # env_snapshot = self.env.snapshot()
            # args[formal_ast.get("name")] = LazyValue(actual_ast, env_snapshot)

        # then create the new activation record 
        self.env.push_func()
        # and add the formal arguments to the activation record
        for arg_name, value in args.items():
          self.env.create(arg_name, value)
        _, return_val = self.__run_statements(func_ast.get("statements"))
        self.env.pop_func()
        return return_val

    def __call_print(self, args):
        output = ""
        for arg in args:
            result = self._eval_expr(arg)  # result is a Value object
            # resolve ANY lazy val here!
            result = self.eval_asap(result)
            output = output + get_printable(result)
        super().output(output)
        return Interpreter.NIL_VALUE

    def __call_input(self, name, args):
        if args is not None and len(args) == 1:
            result = self._eval_expr(args[0])
            # eager eval for input funcs
            result = result.evaluate(self)
            super().output(get_printable(result))

        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        inp = super().get_input()
        if name == "inputi":
            return Value(Type.INT, int(inp))
        if name == "inputs":
            return Value(Type.STRING, inp)

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        expr_ast = assign_ast.get("expression")

        env_snapshot = self.env.snapshot()
        value_obj = LazyValue(expr_ast, env_snapshot)
        
        if not self.env.set(var_name, value_obj):
            super().error(ErrorType.NAME_ERROR, f"Undefined variable {var_name} in assignment")
    
    def __var_def(self, var_ast):
        var_name = var_ast.get("name")
        if not self.env.create(var_name, Interpreter.NIL_VALUE):
            super().error(
                ErrorType.NAME_ERROR, f"Duplicate definition for variable {var_name}"
            )

    def _eval_expr(self, expr_ast):
        try:
            if isinstance(expr_ast, LazyValue): # handle lazy eval
                return expr_ast.evaluate(self)
            if expr_ast.elem_type == InterpreterBase.NIL_NODE:
                return Interpreter.NIL_VALUE
            if expr_ast.elem_type == InterpreterBase.INT_NODE:
                return Value(Type.INT, expr_ast.get("val"))
            if expr_ast.elem_type == InterpreterBase.STRING_NODE:
                return Value(Type.STRING, expr_ast.get("val"))
            if expr_ast.elem_type == InterpreterBase.BOOL_NODE:
                return Value(Type.BOOL, expr_ast.get("val"))
            if expr_ast.elem_type == InterpreterBase.VAR_NODE:
                var_name = expr_ast.get("name")
                val = self.env.get(var_name)
                if val is None:
                    super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
                return val
            if expr_ast.elem_type == InterpreterBase.FCALL_NODE:
                return self.__call_func(expr_ast) # dont call func eagerly during eval expr, and instead save as lazy valye!
                # env_snapshot = self.env.snapshot() # this part didnt work
                # return LazyValue(expr_ast, env_snapshot)
            if expr_ast.elem_type in Interpreter.BIN_OPS:
                return self.__eval_op(expr_ast)
            if expr_ast.elem_type == Interpreter.NEG_NODE:
                return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x)
            if expr_ast.elem_type == Interpreter.NOT_NODE:
                return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)
        except Exception as e: # prop excep for try and catch in eval expr
            raise InterpreterException(str(e))
    
    def eval_asap(self, value):
        if isinstance(value, LazyValue):
            return value.evaluate(self)
        return value

    def __eval_op(self, arith_ast):
        # order of eval must be LEFT -> right
        left_value_obj = self._eval_expr(arith_ast.get("op1"))
        if isinstance(left_value_obj, LazyValue):
            left_value_obj = self.eval_asap(left_value_obj)

        # SHORT CIRCUITING -> just eval the LEFT ONE FIRST!
        if arith_ast.elem_type == "&&":
            if not left_value_obj.value(): # left op is false -> return false
                return Value(Type.BOOL, False)
            right_value_obj = self._eval_expr(arith_ast.get("op2")) # else eval the right one
            if isinstance(right_value_obj, LazyValue):
                right_value_obj = self.eval_asap(right_value_obj)
            return Value(Type.BOOL, right_value_obj.value())

        if arith_ast.elem_type == "||":
            if left_value_obj.value(): # if left is TRUE -> just return true without eval the right!
                return Value(Type.BOOL, True)
            right_value_obj = self._eval_expr(arith_ast.get("op2")) # if false -> do the right!
            if isinstance(right_value_obj, LazyValue):
                right_value_obj = self.eval_asap(right_value_obj)
            return Value(Type.BOOL, right_value_obj.value())

        # continue with evaluating the right operatin!
        right_value_obj = self._eval_expr(arith_ast.get("op2"))
        if isinstance(right_value_obj, LazyValue):
            right_value_obj = self.eval_asap(right_value_obj)

        # division by 0 exception -> raise the special class exception
        if arith_ast.elem_type == "/" and right_value_obj.value() == 0:
            raise InterpreterException("div0")

        # check compatibilities!    
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
        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
        return f(left_value_obj, right_value_obj)

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f):
        value_obj = self._eval_expr(arith_ast.get("op1"))
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

        # eager EVAL: IF
        if isinstance(result, LazyValue): # if lazy, eval ASAP!
            result = result.evaluate(self)

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
        run_for = Interpreter.TRUE_VALUE
        while run_for.value():
            run_for = self._eval_expr(cond_ast)  # check for-loop condition

            # eager EVAL: for
            if isinstance(run_for, LazyValue):
                run_for = run_for.evaluate(self)

            if run_for.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for for condition",
                )
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
        return (ExecStatus.RETURN, value_obj)
