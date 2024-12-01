from interpreterv4 import Interpreter # this may vary

program = """
func bar(x) {
 print("bar: ", x);
 return x;
}

func main() {
 var a;
 a = bar(0);
 a = a + bar(1);
 a = a + bar(2);
 a = a + bar(3);
 print("---");
 print(a);
 print("---");
 print(a);
}

"""

interpreter = Interpreter()
interpreter.run(program) 