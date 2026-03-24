import pylab
import numpy
from scipy.optimize import curve_fit


Directory='/home/studentelab2/benetti_solinas_tedde/Esperienza_13/' # <<<<<< now looking at a file in the datifit directory
NomeFile = 'Oscillatore_forzato_RLC.txt'   # <<<<<< now looking at a file called data00.txt
Filename=(Directory+NomeFile)
# data load
# x,Dx,y,Dy=pylab.loadtxt(Filename,unpack=True)   <<<<< the file is assumed to have 4 columns
f, V_out, d_out, V_in, d_in =pylab.loadtxt(Filename,unpack=True)

G = V_out/V_in

# scatter plot with error bars
pylab.errorbar(f, G, linestyle = 'none', color = 'black', marker = '.')

# bellurie
pylab.rc('font',size=16)
pylab.ylabel('Gain ',fontsize=18)
pylab.xlabel('f  [Hz]',fontsize=18)
pylab.minorticks_on()


# AT THE FIRST STEP (data plot only) YOU MUST COMMENT FROM HERE TO THE LAST LINE (pylab.show())

# make the array with initial values (to be carefully adjusted!)
init=(4.56e-4,2.75e-7,758.74)

# set the error (to be modified if effective errors have to be accounted for)
sigma_G=G*np.sqrt((d_in/V_in)**2+(d_out/V_out)**2)
w=1/sigma_G**2

# define the model function (a straight line in this example)
# note how parameters are entered
# note the syntax
def ff(f, a, b,c):
    return a*f/np.sqrt(b*f**2 + (1-(f/c)**2)**2)

# AT THE SECOND STEP (plot of the model with initial parameters):
# YOU MUST COMMENT FROM HERE TO THE THIRD TO LAST LINE
# (AND PUT IN THAT LINE *init IN THE PLACE OF *pars)
# call the routine
pars,covm=curve_fit(ff,f,G,init,sigma_G,absolute_sigma=True) # <<<< NOTE THE absolute_sigma option

# calculate the kappasquare for the best-fit funtion
# note the syntax for the pars array
kappa2 = ((w*(G-ff(f,*pars))**2)).sum()

# determine the ndof
ndof=len(f)-len(pars)

# print results on the console
print(pars)
print(np.sqrt(covm.diagonal()))
print (kappa2, ndof)


# AT THE SECOND STEP, COMMENT UP TO HERE
# prepare a dummy xx array (with 500 linearly spaced points)
xx=numpy.linspace(min(f),max(f),500)

# plot the fitting curve with either the initial or the optimised parameters
# AT THE SECOND STEP, YOU MUST REPLACE *pars WITH *init
pylab.plot(xx,ff(xx,*pars), color='red')


# show the plot
pylab.show()