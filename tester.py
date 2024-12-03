from interpreterv4 import Interpreter # this may vary

program = """
func main() {
 var a;
 var b;
 a = 10;
 b = a + 1;
 a = a + 10;
 b = b + a;
 print(a);
 print(b);
}
"""

interpreter = Interpreter()
interpreter.run(program) 