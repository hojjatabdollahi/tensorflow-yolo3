import csv
import os
import random


input_file = 'original.csv'
output_file = 'original25k.csv'
eval_file = 'eval400.csv'

def create_train():
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
        for _ in range(75000):
            index = random.randint(0,len(output_lines)-1)
            output.write(output_lines[index])

def create_eval():
    # read all 230k lines
    lines = []
    with open(input_file, 'r') as ifile:
        lines = ifile.readlines()

    output_lines = []
    train_counters = [25000,25000,25000]
    eval_counters = [400,400,400]

    # make sure to skip the first 25k of each class and just get 400
    for line in lines:
        index = int(line.split(',')[1].replace('"', ''))-1
        if train_counters[index] != 0:
            train_counters[index] -= 1
        elif eval_counters[index] != 0:
            eval_counters[index] -= 1
            output_lines.append(line)


    print(len(output_lines))

    # Randomize
    with open(eval_file,'w') as output:
        for _ in range(len(output_lines)):
            index = random.randint(0,len(output_lines)-1)
            output.write(output_lines[index])

create_eval()
