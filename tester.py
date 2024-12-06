from interpreterv4 import Interpreter # this may vary

program = """
func a(x) {
  print("a() running");
  return x;
}

func main() {
  var x;
  var y;
  x = true || false;
  y = inputi(a(x));
  print("y assigned");
  print(y);
}

"""

interpreter = Interpreter()
interpreter.run(program) 
