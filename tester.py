from interpreterv4 import Interpreter # this may vary

program = """
func foo(a, b, c, d) {
  print(a);
  print(c);
  return "all good";
}

func get_num() {
  print("get num");
  return 5 * 6;
}

func get_str() {
  print("get string");
  return "5 * 6";
}

func get_bool() {
  print("get bool");
  return !false;
}

func get_error1() {
  print("should not get called");
  var a;
  var a;
}

func get_error2() {
  print("should not get called");
  undefinedvar = 5;
}

func raise_error() {
  print("should not get called");
  raise "err";
}

func main() {
  var x;
  var y;
  x = foo(get_num(), get_error2(), get_bool(), does_not_exist());
  y = foo(get_str(), raise_error(), get_num(), get_error1());

  print(y);
  print(x);
}

"""

interpreter = Interpreter()
interpreter.run(program) 
