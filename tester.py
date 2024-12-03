from interpreterv4 import Interpreter # this may vary

program = """
func foo() {
  print("foo");
  return 4;
}

func main() {
  foo();
  print("---");
  var x;
  x = foo();
  print("---");
  print(x); 
}

"""

interpreter = Interpreter()
interpreter.run(program) 