#"(1)"
# initialize
m1 = [3,1]
c1 = [[1,2], [2,5]]
m2 = [1,3]
c2 = [[1,2], [2,5]]

# import numpy and pyplot
import numpy as np
import matplotlib.pyplot as plt

# generate 50 random numbers by using multivariate_normal
n = 50
x1 = np.random.multivariate_normal(m1, c1, n)
x2 = np.random.multivariate_normal(m2, c2, n)

# plot x1 and x2
plt.plot(x1[:, 0], x1[:, 1], 'o')
plt.plot(x2[:, 0], x2[:, 1], 'o')
plt.xlim(-2, 6)
plt.ylim(-2.5, 5)
plt.show()




#"(2)"

#impot linalg
import numpy.linalg as LA

# to apply PCA, concatenate x1 and x2 to x.
x = np.concatenate([x1,x2])
# get mean and covariance matrix
m = np.mean(x, axis=0)
c = np.cov(x, rowvar=0, bias=0)

# get eignvaue and eignvector
eignvalue, eignvector = LA.eig(c)
# get the index where eignvalue is max, and it's vector
index_maxeignvalue = np.where(eignvalue==max(eignvalue))[0][0]
eignvector_maxeignvalue = eignvector[:, index_maxeignvalue]

# to plot the boudary, get points by using mean and eignvector 
xlist = np.arange(-3, 8)
ylist = m[1] + (eignvector_maxeignvalue[1]/eignvector_maxeignvalue[0]) * (xlist - m[0])

# plot
plt.plot(xlist, ylist.real)
plt.plot(x1[:, 0], x1[:, 1], 'o')
plt.plot(x2[:, 0], x2[:, 1], 'o')
plt.xlim(-2, 6)
plt.ylim(-2.5, 5)
plt.show()





#"(3)"

# within-class scatters and sum
s1 = n * np.cov(x1, rowvar=0, bias = 0)
s2 = n * np.cov(x2, rowvar=0, bias = 0)
sw = s1 + s2
sw = np.array(sw)

# get each mean
m11 = np.mean(x1, axis=0)
m22 = np.mean(x2, axis=0)

# get A optimizing LDA
A = LA.inv(sw) @ (m11 - m22)

# to plot the boudary, get points by using mean and A 
xlist = np.arange(-3, 8)
ylist = m[1] + (A[1]/A[0]) * (xlist - m[0])

# plot
plt.plot(xlist, ylist.real)
plt.plot(x1[:, 0], x1[:, 1], 'o')
plt.plot(x2[:, 0], x2[:, 1], 'o')
plt.xlim(-2, 6)
plt.ylim(-2.5, 5)
plt.show()





#"(4)"

PCAvalues = []
LDAvalues = []
for i in x:
  PCAvalues.append(eignvector_maxeignvalue @ i)
  LDAvalues.append(A @ i)

plt.subplot(1,2,1)
plt.hist(PCAvalues)
plt.title("PCA")

plt.subplot(1,2,2)
plt.hist(LDAvalues)
plt.title("LDA")

plt.tight_layout()
plt.show()
