'''
Utility functions for plotting
'''
import matplotlib.pyplot as plt


def plot(data):
    '''
    Plot data
    '''
    epochs = []
    accuracy = []
    parameters = []
    times = []

    for epoch, data_point in enumerate(data):
        epochs.append(epoch)
        for pair in data_point:
            key = pair.keys()[0]
            if(key == 'final_acc'):
                val = pair[key]
                accuracy.append(val)
            elif(key == 'params'):
                val = pair[key]
                parameters.append(val)
            elif(key == 'final_time'):
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
