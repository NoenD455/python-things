calc = 1

if 1 == 2:
    input1 = input()
    input2 = input()
    input3 = input()
1
if calc == 1:
    print("first number")
    input1 = input()
    calc += 1

if calc == 2:
    print("function")
    input2 = input()
    calc += 1

if calc == 3:
    print("second number")
    input3 = input()
    calc +=1

if calc == 4:
    if input2 == "+":
        print(int(input1)+int(input3))
    if input2 == "*":
        print(int(input1)*int(input3))
    if input2 == "/":
        print(int(input1)/int(input3))
    if input2 == "-":
        print(int(input1)-int(input3))