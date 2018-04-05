import nltk
from Bio import SeqIO
from functools import reduce
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import sys
import glob
import os 
import csv
from sklearn.metrics import precision_recall_curve

#======== THIS IS FOR PARSING TEST SET WITH PCFG===========#

#========EVALUATION======#


#function to read data from FASTA file, returns array with sentences
def read_val_data(file_name):
    sentences = []
    with open(file_name) as file:
        for seq_record in SeqIO.parse(file,'fasta'):
            sentences.append((str(seq_record.seq)))
    return sentences


#parsing with Viterbi algorithm using grammar specified file data 
def parse(file, labelclass, grammar):
    gram = nltk.PCFG.fromstring(open(grammar,"r"))#load grammar file as PCFG class
    viterbi_parser = nltk.ViterbiParser(gram) #set the parser
    sentences = read_val_data(file)
    
    props = []
    if (labelclass == 1):
        label = np.ones(len(sentences))
    else:
        label = np.zeros(len(sentences))
    for sentence in sentences:
        sentence = list(sentence.lower())
        parses = viterbi_parser.parse_all(sentence)
        average = (reduce(lambda a, b:a+b.prob(), parses, 0)/len(parses) if parses else 0)
        props.append(average)
    props = np.array(props)
    props_final = np.vstack((props, label))
    return props_final

#get fold path:
path = sys.argv[1]

#find PCFG in fold folder
for fileg in glob.glob(os.path.join(path, "*.pcfg")):
    print("I found")
    grammar = fileg

#find positive data and parse it
for file in glob.glob(os.path.join(path, "hets*val.fa")):
    positives = parse(file, 1,grammar)
print("OK positives")

#find negative data and parse it
for file in glob.glob(os.path.join(path, "negsample*val.fa")):
    negatives=parse(file,0,grammar)
print("OK negatives")

#build the dataset from annotated data
result=np.hstack((positives,negatives))
print("OK data")

#read the threshold for this fold

#number of fold
number=int(path[-2:])-1

#read threshold file

thr=[]
with open('max_f1_thr.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        thr.append(row)

#find threshold for this fold

current_thr = float(thr[number][0])

#use the threshold to predict

label = result[1,:]
scores=result[0,:]

prediction=[]
for x in scores:
    if x >= current_thr:
        lab=1
    else:
        lab=0
    prediction.append(lab)

whole=np.vstack((scores, label, prediction))

print(sklearn.metrics.classification_report(label, prediction))

tn, fp, fn, tp = sklearn.metrics.confusion_matrix(label, prediction).ravel()
precision=sklearn.metrics.precision_score(label, prediction)
recall=sklearn.metrics.recall_score(label, prediction)
f1= sklearn.metrics.f1_score(label, prediction) 
result=[tn, tp, fn, fp, precision, recall, f1]



with open(r'labels.csv', 'a') as file:
    writer=csv.writer(file)
    writer.writerow(label)

with open(r'prediction.csv', 'a') as file:
    writer=csv.writer(file)
    writer.writerow(prediction)


with open(r'result.csv', 'a') as file:
    writer=csv.writer(file)
    writer.writerow(result)

