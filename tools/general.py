'''
Commonly used functions 

'''

def standard_error(x):
    return x.std() / (len(x) ** 0.5)

def jackknife_standard_error(x):
    return x.std() * (len(x) ** 0.5)

