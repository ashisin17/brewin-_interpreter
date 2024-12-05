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

"""
func main() {
 var b;
 b = 5; 
 - acces b
 - b = new val with int: val: 5
var a;
a = b + 1;  
- lazy val created for b+1 and a is updated
- b+1 deferred!
b = 10;
- new value of b created with b = 10
print(a);
1. access a
2. access
}
"""