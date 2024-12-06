from interpreterv4 import Interpreter # this may vary

program = """
func foo() {
  try {
    raise "z";
  }
  catch "x" {
    print("x");
  }
  catch "y" {
    print("y");
  }
  catch "z" {
    print("z");
    raise "a";
  }
  print("q");
}

func main() {
  try {
    foo();
    print("b");
  }
  catch "a" {
    print("a");
  }
}

"""

interpreter = Interpreter()
interpreter.run(program) 
