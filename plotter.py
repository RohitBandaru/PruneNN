import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load( open( "data_124_3epochs_5thres.pkl", "rb" ) )

epochs = []
accuracy = []
parameters = []
times = []
flops = []
thres = []

for epoch in range(len(data)):
    data_point = data[epoch]
    if(len(data_point) == 0):
        break
    epochs.append(epoch)
    for pair in data_point:
        key = pair.keys()[0]
        if(key=='final_acc'):
            val = pair[key]
            accuracy.append(val)
        elif(key=='params'):
            val = pair[key]
            parameters.append(val)
        elif(key=='final_time'):
            val = pair[key]
            times.append(val)

plt.plot(epochs, accuracy)
plt.title('accuracy')
plt.show()

plt.plot(epochs, parameters)
plt.title('parameters')
plt.show()

plt.plot(epochs, times)
plt.title('times')
plt.show()
