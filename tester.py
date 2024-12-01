from interpreterv4 import Interpreter # this may vary

program = """
func main(): void {
    var x: int;
    print(x == nil);
}
"""

interpreter = Interpreter()
interpreter.run(program) 