import numpy as np

def data():
    data = []
    for i in xrange(1000):
        data.append([i,i,i])
    return np.asarray(data)

if __name__ == '__main__':
    np.savetxt("Data/best4linreg.csv", data(), delimiter=',')
