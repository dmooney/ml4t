import numpy as np

def data():
    data = []
    for i in xrange(250):
        data.append([0, 0, -11])
        data.append([1332, -123121, 37])
        data.append([-123890723, -96968, 400])
        data.append([1, 939, -1000000])
    return np.asarray(data)

if __name__ == '__main__':
    np.savetxt("Data/best4KNN.csv", data(), delimiter=',')
