from interpreterv4 import Interpreter # this may vary

program = """
func foo() {
 print("foo");
 return true;
}

func bar() {
 print("bar");
 return false;
}

func main() {
  print(foo() || bar() || foo() || bar());
  print("done");
}
"""

interpreter = Interpreter()
interpreter.run(program) 
