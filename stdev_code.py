import math as m

# calculate the population standard deviation
def stdev_p(data):
    mean = m.fsum(data) / len(data)
    deviations = [(x - mean)**2 for x in data]
    variance = m.fsum(deviations) / len(deviations)
    result = m.sqrt(variance)
    return result

# calculate the sample standard deviation
def stdev_s(data):
    mean = m.fsum(data) / len(data)
    deviations = [(x - mean)**2 for x in data]
    variance = m.fsum(deviations) / (len(deviations) - 1)
    result = m.sqrt(variance)
    return result

if __name__ == "__main__":
    test = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    print "the population stdev is", stdev_p(test)
    print "the sample stdev is", stdev_s(test)

    test2 = [60.0, 16.0, 34.0, 84.0, 46.0, 53.0, 87.0, 72.0, 39.0, 60.0, 20.0, 63.0, 27.0, 7.0, 43.0, 90.0, 100.0, 97.0, 5.0, 14.0, 18.0, 47.0, 81.0, 93.0, 89.0, 2.0, 85.0, 13.0, 3.0, 90.0, 32.0, 93.0, 9.0, 46.0, 8.0, 78.0, 88.0, 34.0, 11.0, 53.0, 22.0, 53.0, 2.0, 62.0, 71.0, 76.0, 74.0, 49.0, 8.0, 73.0, 80.0, 83.0, 31.0, 34.0, 18.0, 67.0, 49.0, 34.0, 82.0, 5.0, 84.0, 53.0, 54.0, 20.0, 2.0, 99.0, 62.0, 19.0, 46.0, 62.0, 29.0, 57.0, 60.0, 68.0, 30.0, 73.0, 85.0, 47.0, 92.0, 75.0, 54.0, 36.0, 13.0, 68.0, 13.0, 16.0, 51.0, 42.0, 27.0, 54.0, 65.0, 97.0, 50.0, 69.0, 43.0, 49.0, 42.0, 4.0, 31.0, 73.0]
    print "the population stdev is", stdev_p(test2)
    print "the sample stdev is", stdev_s(test2)
