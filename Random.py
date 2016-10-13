from numpy import *
import pandas as pd
import csv

df = pd.read_csv('train3.csv')
data = df.values[1:1941, 0:].tolist()
random.shuffle(data)
csvfile = file('csvtest.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerows(data)
csvfile.close()