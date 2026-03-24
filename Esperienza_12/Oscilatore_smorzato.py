import pylab
import numpy as np
from scipy.optimize import curve_fit


Directory='/home/studentelab2/benetti_solinas_tedde/Esperienza_12/' # <<<<<< now looking at a file in the datifit directory
NomeFile = 'Pasqua.04_M_CH1.txt'   # <<<<<< now looking at a file called data00.txt
Filename=(Directory+NomeFile)
# data load
t,Dt,V,DV=pylab.loadtxt(Filename,unpack=True)   #the file is assumed to have 4 columns
#vin, vout, f =pylab.loadtxt(Filename,unpack=True)

#G = vout/vin

# scatter plot with error bars
pylab.errorbar(t, V, xerr=Dt, yerr=DV, linestyle = 'none', color = 'black', marker = '.')

# bellurie
pylab.rc('font',size=16)
pylab.xlabel('t [s]',fontsize=18)
pylab.ylabel('V  [digit]',fontsize=18)
pylab.minorticks_on()


# AT THE FIRST STEP (data plot only) YOU MUST COMMENT FROM HERE TO THE LAST LINE (pylab.show())

# make the array with initial values (to be carefully adjusted!)
init=(130,9e3,10e-3,1.5,0)

# set the error (to be modified if effective errors have to be accounted for)
sigma=DV
s=1/sigma**2

# define the model function (a straight line in this example)
# note how parameters are entered
# note the syntax
def ff(t, A, w, tau, phi,offset):
    return A*np.exp(-t/tau)*np.cos(w*t + phi)+offset

# AT THE SECOND STEP (plot of the model with initial parameters):
# YOU MUST COMMENT FROM HERE TO THE THIRD TO LAST LINE
# (AND PUT IN THAT LINE *init IN THE PLACE OF *pars)
# call the routine

popt,pcov=curve_fit(ff,t,V,init,sigma,absolute_sigma=True) # <<<< NOTE THE absolute_sigma option
A, w, tau, phi,offset=popt
sigma_A, sigma_w, sigma_tau, sigma_phi,sigma_offset=np.sqrt(np.diag(pcov))
print(A, w, tau, phi,offset)
print(sigma_A, sigma_w, sigma_tau, sigma_phi,sigma_offset)
# calculate the kappasquare for the best-fit funtion
# note the syntax for the pars array
kappa2 = ((s*(V-ff(t,*popt))**2)).sum()

# determine the ndof
ndof=len(t)-len(popt)

# print results on the console
# print(pars)
# print(covm)
print (kappa2, ndof)


# AT THE SECOND STEP, COMMENT UP TO HERE
# prepare a dummy xx array (with 500 linearly spaced points)
xx=np.linspace(min(t),max(t),500)

# plot the fitting curve with either the initial or the optimised parameters
# AT THE SECOND STEP, YOU MUST REPLACE *pars WITH *init
pylab.plot(t,ff(t,*popt), color='red')


# show the plot
pylab.show()