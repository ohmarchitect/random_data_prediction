#dk# print(random.randint(0,9))
#dk# number_of_threads = (random.randint(1,10))
#dk# range_in_time     = (random.randint(1,1000))
#dk# print "Number of threads = " + str(number_of_threads)
#dk# print "Range in time     = " + str(range_in_time)
#dk#
#dk#
#dk# # Create initial random data
#dk# threads = []
#dk# for thread in range(0, number_of_threads):
#dk#    current_thread = []
#dk#    random_iteration_change = random.randint(0,100)
#dk#
#dk#    incr0_decr1_same2_3mult_4div = random.randint(0,2)
#dk#    for value in range(0, range_in_time):
#dk#       if random.randint(0,10) == random.randint(0,10):
#dk#          incr0_decr1_same2_3mult_4div = random.randint(0,2)
#dk#       if value == 0:
#dk#          current_thread.append(random.randint(0,100))
#dk#       elif incr0_decr1_same2_3mult_4div == 0:
#dk#          current_thread.append(random.randint(0,100) + current_thread[-1])
#dk#       elif incr0_decr1_same2_3mult_4div == 1:
#dk#          current_thread.append(random.randint(0,100) - current_thread[-1])
#dk#       elif incr0_decr1_same2_3mult_4div == 2:
#dk#          current_thread.append(random.randint(0,100))
#dk#       elif incr0_decr1_same2_3mult_4div == 3:
#dk#          current_thread.append(random.randint(0,100)*current_thread[-1])
#dk#       elif incr0_decr1_same2_3mult_4div == 4:
#dk#          current_thread.append(random.randint(0,100)/current_thread[-1])
#dk#
#dk#    threads.append(current_thread)
#dk#
#dk#
#dk# t = numpy.arange(0, range_in_time, 1)
#dk#
#dk#
#dk# # Create causality
#dk# number_of_related_threads = (random.randint(1,number_of_threads))
#dk# for relation in range(0,number_of_related_threads):
#dk#    tx_thread = random.randint(0,number_of_threads-1)
#dk#    rx_thread = random.randint(0,number_of_threads-1)
#dk#    print "Thread " + str(tx_thread) + " feeds into thread " + str(rx_thread)
#dk#
#dk#    new_threads = []
#dk#    for thread in range(0, number_of_threads):
#dk#       current_thread = []
#dk#       random_iteration_change = random.randint(0,100)
#dk#
#dk#       incr0_decr1_same2_3mult_4div = random.randint(0,4)
#dk#       for value in range(0, range_in_time):
#dk#
#dk#          if thread == rx_thread:
#dk#             if random.randint(0,random_iteration_change) == random.randint(0,random_iteration_change):
#dk#                incr0_decr1_same2_3mult_4div = random.randint(0,2)
#dk#             if incr0_decr1_same2_3mult_4div == 0:
#dk#                current_thread.append(random.randint(1,100) + threads[tx_thread][value])
#dk#             elif incr0_decr1_same2_3mult_4div == 1:
#dk#                current_thread.append(random.randint(1,100) - threads[tx_thread][value])
#dk#             elif incr0_decr1_same2_3mult_4div == 2:
#dk#                current_thread.append(random.randint(1,100))
#dk#             elif incr0_decr1_same2_3mult_4div == 3:
#dk#                current_thread.append(random.randint(1,100)*threads[tx_thread][value])
#dk#             elif incr0_decr1_same2_3mult_4div == 4:
#dk#                current_thread.append(random.randint(1,100)/threads[tx_thread][value])
#dk#          else:
#dk#             current_thread.append(threads[thread][value])
#dk#       new_threads.append(current_thread)


import random
import matplotlib.pyplot as pyplot
import numpy
import scipy
from scipy.fftpack import fft
from sklearn import preprocessing

def random_market(t):
   market = [];
   market.append(random.random() )
   volatility = random.random()
   for x in t:
      old_price = market[-1]
      rnd = random.random() ; # generate number, 0 <= x < 1.0
      change_percent = 2 * volatility * rnd;
      if (change_percent > volatility):
          change_percent -= (2 * volatility);
      change_amount = old_price * change_percent;
      market.append(old_price + change_amount);
   return market

def cosine_similarity(a, b):
    return numpy.dot(a, b) / numpy.linalg.norm(a) / numpy.linalg.norm(b)

t = range(0,100)
rx_signal, tx_signal = preprocessing.normalize([random_market(t), random_market(t)])
t.append(101)

output_signal = []
for value in range(0,len(tx_signal)):
    output_signal.append(tx_signal[value] + rx_signal[value])

pyplot.figure(1)
pyplot.subplot(3,2,1)
pyplot.plot(t, rx_signal)

input_f = (fft(rx_signal))
pyplot.subplot(3,2,2).set_yscale('log')
pyplot.plot(t[0:len(t)/2], numpy.abs(input_f[0:len(t)/2]))


pyplot.subplot(3,2,3)
pyplot.plot(t, tx_signal)

factor_f = (fft(tx_signal))
pyplot.subplot(3,2,4).set_yscale('log')
pyplot.plot(t[0:len(t)/2], numpy.abs(factor_f[0:len(t)/2]))


pyplot.subplot(3,2,5)
pyplot.plot(t, output_signal)

output_f = (fft(output_signal))
pyplot.subplot(3,2,6).set_yscale('log')
pyplot.plot(t[0:len(t)/2], numpy.abs(output_f[0:len(t)/2]))


from scipy import spatial
print "--- cosine similarity  --------"
print cosine_similarity(tx_signal, rx_signal)
print cosine_similarity(tx_signal, output_signal)
print cosine_similarity(rx_signal, output_signal)
print "--- cosine similarity of fft --------"
print cosine_similarity(abs(input_f),  abs(factor_f))
print cosine_similarity(abs(input_f),  abs(output_f))
print cosine_similarity(abs(factor_f), abs(output_f))
print "--- spatial distance --------"
print spatial.distance.cdist([tx_signal,tx_signal], [rx_signal    ,rx_signal    ])
print spatial.distance.cdist([tx_signal,tx_signal], [output_signal,output_signal])
print spatial.distance.cdist([rx_signal,rx_signal], [output_signal,output_signal])
print "---- spatial distance of fft -------"
print spatial.distance.cdist([abs(input_f) , abs(input_f) ], [abs(factor_f),abs(factor_f)])
print spatial.distance.cdist([abs(input_f) , abs(input_f) ], [abs(output_f),abs(output_f)])
print spatial.distance.cdist([abs(factor_f), abs(factor_f)], [abs(output_f),abs(output_f)])
print "-----------"

pyplot.show()
exit()



#   threads = new_threads


