def gen(n):
	for i in range(n):
		yield i**2

for i in gen(10000):
	print(i," ",end="")
print("")