# The EnvironmentManager class keeps a mapping between each variable name (aka symbol)
# in a brewin program and the Value object, which stores a type, and a value.
import copy

class EnvironmentManager:
    def __init__(self):
        self.environment = []

    # returns a VariableDef object
    def get(self, symbol):
        cur_func_env = self.environment[-1]
        for env in reversed(cur_func_env):
            if symbol in env:
                value_obj = env[symbol]
                print(f"DEBUG: Accessing variable {symbol}, value: {value_obj}")
                if value_obj.is_lazy():
                    # print(f"DEBUG: Resolving lazy value for variable {symbol}")
                    lazy_resolved = value_obj.evaluate()
                    env[symbol] = lazy_resolved # TODO: dont wanan change the envr? come back!
                    # print(f"DEBUG: Updated variable {symbol} with resolved value: {lazy_resolved}")
                    return lazy_resolved
                return value_obj

        return None

    def set(self, symbol, value):
        cur_func_env = self.environment[-1]
        for env in reversed(cur_func_env):
            if symbol in env:
                env[symbol] = value
                return True

        return False

    # create a new symbol in the top-most environment, regardless of whether that symbol exists
    # in a lower environment
    def create(self, symbol, value):
        cur_func_env = self.environment[-1]
        if symbol in cur_func_env[-1]:   # symbol already defined in current scope
            return False
        cur_func_env[-1][symbol] = value
        return True

    # used when we enter a new function - start with empty dictionary to hold parameters.
    def push_func(self):
        self.environment.append([{}])  # [[...]] -> [[...], [{}]]

    def push_block(self):
        cur_func_env = self.environment[-1]
        cur_func_env.append({})  # [[...],[{....}] -> [[...],[{...}, {}]]

    def pop_block(self):
        cur_func_env = self.environment[-1]
        cur_func_env.pop() 

    # used when we exit a nested block to discard the environment for that block
    def pop_func(self):
        self.environment.pop()
    
    # LAZY eval func
    def snapshot(self): 
        # create SHALLOW copy with the obj ref!
        snapshot_env = []
        for func_env in self.environment:
            snapshot_func_env = [{key: value for key, value in block.items()} for block in func_env]
            snapshot_env.append(snapshot_func_env)

        snapshot_manager = EnvironmentManager()
        snapshot_manager.environment = snapshot_env
        return snapshot_manager


