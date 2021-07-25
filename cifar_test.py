from os import listdir
from os.path import isfile, join
mypath= 'C:\Go'
# my_path = 'C:\Users\jj\Desktop\analysis'
for f in listdir(mypath):
    if f.endswith('.txt'):
        print(f)