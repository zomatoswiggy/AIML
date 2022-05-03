
from matplotlib.pyplot import plot


year,profit=loadtxt('company-a-data.txt',unpack=True)

scatter(year,profit,color='r',marker='d')

#LogLog plot

loglog(x,y)

x=linspace(1,20,100)
y=5*x**3
clf()

loglog(x,y)
  