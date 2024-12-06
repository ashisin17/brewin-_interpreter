from interpreterv4 import Interpreter # this may vary

program = """
func foo() {
  print("F1");
  raise "except1";
  print("F3");
}

func bar() {
 try {
   print("B1");
   foo();
   print("B2");
 }
 catch "except2" {
   print("B3");
 }
 print("B4");
}

func main() {
 try {
   print("M1");
   bar();
   print("M2");
 }
 catch "except1" {
   print("M3");
 }
 catch "except3" {
   print("M4");
 }
 print("M5");
}
"""

interpreter = Interpreter()
interpreter.run(program) 
