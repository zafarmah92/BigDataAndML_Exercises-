
import sys

print (sys.argv)
print (sys.argv[1])


target_file = target = open('1_2_3_4.txt', 'w')

for i in (range(0,len(sys.argv)-1)):
	file = sys.argv[i+1]
	
	print (file)

	with open(file,'rb') as f:
		#print (f.read())
		_string = f.read()
		_string = str(_string)
		print(_string)
		target.write(_string)
		target.write("\n\n\n\nnext_file\n\n\n\n")


target_file.close()
