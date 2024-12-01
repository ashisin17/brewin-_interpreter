from interpreterv4 import Interpreter # this may vary

program = """
func zero() {
  print("zero");
  return 0;
}

func inc(x) {
 print("inc:", x);
 return x + 1;
}

func main() {
 var a;
 for (a = 0; zero() + a < 3; a = inc(a)) {
   print("x");
 }
 print("d");
}
"""

interpreter = Interpreter()
interpreter.run(program) 