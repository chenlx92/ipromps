#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_data = pd.read_csv('./multiModal_states.csv')
plt.figure(0) # right hand
plt.plot(range(len(all_data.values)), all_data.values[:,240])
plt.plot(range(len(all_data.values)), all_data.values[:,241])
plt.plot(range(len(all_data.values)), all_data.values[:,242])

plt.figure(1) # left hand 
plt.plot(range(len(all_data.values)), all_data.values[:,207])
plt.plot(range(len(all_data.values)), all_data.values[:,208])
plt.plot(range(len(all_data.values)), all_data.values[:,209])

plt.figure(2) # foot 
plt.plot(range(len(all_data.values)), all_data.values[:,273])
plt.plot(range(len(all_data.values)), all_data.values[:,274])
plt.plot(range(len(all_data.values)), all_data.values[:,275])

plt.figure(3) # foot 
plt.plot(range(len(all_data.values)), all_data.values[:,306])
plt.plot(range(len(all_data.values)), all_data.values[:,307])
plt.plot(range(len(all_data.values)), all_data.values[:,308])

plt.figure(4) # knee 
plt.plot(range(len(all_data.values)), all_data.values[:,295])
plt.plot(range(len(all_data.values)), all_data.values[:,296])
plt.plot(range(len(all_data.values)), all_data.values[:,297])

plt.figure(5) # right elbow 
plt.plot(range(len(all_data.values)), all_data.values[:,229])
plt.plot(range(len(all_data.values)), all_data.values[:,230])
plt.plot(range(len(all_data.values)), all_data.values[:,231])

plt.figure(6) # left elbow 
plt.plot(range(len(all_data.values)), all_data.values[:,196])
plt.plot(range(len(all_data.values)), all_data.values[:,197])
plt.plot(range(len(all_data.values)), all_data.values[:,198])

print(all_data.values[0,208]) #right_hand_1