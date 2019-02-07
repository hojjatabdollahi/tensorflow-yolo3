import csv
import os
import random


input_file = '/media/data/hojjat_data/affectnet/original.csv'
output_file = '/media/data/hojjat_data/affectnet/original25k.csv'

# read all 230k lines
lines = []
with open(input_file, 'r') as ifile:
    lines = ifile.readlines()

counters = [25000,25000,25000]

output_lines = []

# make sure just get 25k of each class
for line in lines:
    index = int(line.split(',')[1].replace('"', ''))-1
    if counters[index] != 0:
        counters[index] -= 1
        output_lines.append(line)

print(len(output_lines))

# Randomize
with open(output_file, 'w') as output:
    for i in range(75000):
        index = random.randint(0,len(output_lines)-1)
        output.write(output_lines[index])
