from interpreterv4 import Interpreter # this may vary

program = """
func main() {
  var result;
  var x;
  x = 3;
  result = x + 10;
  x = 4;
  print(result);  
}
"""

interpreter = Interpreter()
interpreter.run(program) 