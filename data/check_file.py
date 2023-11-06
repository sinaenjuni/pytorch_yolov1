train_file = open("./train.txt", 'r').read().strip().split()
val_file = open("./val.txt", 'r').read().strip().split()
test_file = open("./test.txt", 'r').read().strip().split()

# print(train_file)

dif1 = list(set(train_file) - set(test_file))
dif2 = list(set(test_file) - set(train_file))

# 두 파일중 공통된 요소 출력 
print(list((set(test_file)-set(dif1))-set(dif2)))


dif1 = list(set(train_file) - set(val_file))
dif2 = list(set(val_file) - set(train_file))

# 두 파일중 공통된 요소 출력 
print(list((set(val_file)-set(dif1))-set(dif2)))