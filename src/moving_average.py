import numpy
import matplotlib.pyplot as plt

def RandomWalk(N=1000, d=2):
    """
    Use numpy.cumsum and numpy.random.uniform to generate
    a 2D random walk of length N, each of which has a random DeltaX and
    DeltaY between -1/2 and 1/2.  You'll want to generate an array of 
    shape (N,d), using (for example), random.uniform(min, max, shape).
    """
    return numpy.cumsum(numpy.random.uniform(-0.5,0.5,(N,d)))
    
def movingaverage(data, window_width):
    cumsum_vec = numpy.cumsum(numpy.insert(data, 0, 0)) 
    x_ma = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    x_ma = x_ma.tolist()
    x_ma = [None]*(len(data)-len(x_ma)) + x_ma
    y_ma = range(0,len(x_ma))
    return x_ma,y_ma

N=1000
d=2
x = RandomWalk(N,d)
y = range(0,len(x))

x_ma = {}
y_ma = {}
fig,ax = plt.subplots(figsize=(8,4))

resolution = 5
for i in range(1,len(x)-1,resolution):
    x_ma[i], y_ma[i] = movingaverage(x,i)
    print str(i/100)
    ax.plot(y_ma[i], x_ma[i], color=str(float(i)/(N*d)))
    
ax.plot(y,x, color='blue')
ax.grid(True)
ax.legend(loc='best', prop={'size':'large'})
fig.suptitle('Moving Average')
fig.show()