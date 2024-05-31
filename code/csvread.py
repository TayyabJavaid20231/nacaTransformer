import csv
import numpy as np
# File path
my_list128 = 'my_list128.txt'
my_list256 = 'my_list256.txt'

# Initialize an empty list to store the data
data_list_128 = []
data_list_256 = []

# Reading the list from the CSV file
with open(my_list128, 'r') as file:
    for row in file:
        data_list_128.append(row)

with open(my_list256, 'r') as file:
    for row in file:
        data_list_256.append(row)
#print(data_list_128[0])
print(data_list_128[0][2:-3])
print(data_list_128[0][2:-1])

print(data_list_256[0][2:-2])

print(data_list_128[0][2:-2] == data_list_256[0][2:-2])

val = 0
for i in range(len(data_list_128)):
    if data_list_128[i][2:-2] == data_list_256[i][2:-2]:
        val = val + 1

print(val)
