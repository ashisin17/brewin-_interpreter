from interpreterv4 import Interpreter # this may vary

program = """
func bar(x) {
 print("bar: ", x);
 return x;
}

func main() {
 var a;
 a = -bar(1);
 print("---");
 print(a);
}
"""

interpreter = Interpreter()
interpreter.run(program) 