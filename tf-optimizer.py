#from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function

import pandas
import numpy as np
import random
from numpy.polynomial.hermite  import hermgauss
import tensorflow as tf

kT = 1.986e-3*300.0 #1/kT at 300K 
beta = 1./kT

xg,wg = hermgauss(15)
xgauss = np.float32(xg)
wgauss = np.float32(wg)
ngauss = xgauss.size


## Read in data
ulambda = pandas.read_csv("sampl4-oah-b7eqnosc.repl.cycle.totE.potE.temp.lambda.ebind.dat",delim_whitespace=True, 
                          header=None,names=["replica","cycle","totE",
                                             "potE","temp","Lambda","ebind"])

#ulambda = pandas.read_csv("bcd-nabumetone.repl",delim_whitespace=True, 
#                          header=None,names=["replica","cycle","totE",
#                          "potE","temp","Lambda","ebind"])




u = np.float32(ulambda['ebind'])*beta #transform in units of kT
lmbs = np.float32(ulambda['Lambda'])
cycles = np.int32(ulambda['cycle'])

#mask = np.logical_and(lmbs < 0.3 , lmbs > 1.e-3)
mask = np.logical_and(lmbs > 1.e-6, cycles > 100)
#mask = np.abs(lmbs - 0.1) < 1.e-6
un = u[mask]
ln = lmbs[mask]

#initial parameters
ubA = -13.953*beta
sbA = 1.906*beta
ubB = -2.334*beta
sbB = 3.238*beta
pA = 2.45e-4
pB = 1. - pA
elj = 20.0*beta
uc = 0.5*beta
nl = 2.96
pb = 1.18e-4
pc = 3.00e-2
pm = 1. - pb - pc
umax = 1.e6*beta

#integration grid for Pwca
a = uc
ymax = np.log(umax)/a 
dy = ymax/100.0
xasymp = np.float32(np.exp(a*np.arange(0.,ymax,dy)) - 1.) + 1.e-6
dxasymp = np.float32(xasymp[1:] - xasymp[:-1])
nx = dxasymp.size
xasymp = np.float32(xasymp[:-1])

#dictionary of parameters
params = {}

params['UBA'] = tf.constant(ubA)
#params['UBA'] = tf.get_variable("UBA", initializer=tf.constant(ubA))

params['SIGBA'] = tf.constant(sbA)
#params['SIGBA'] = tf.get_variable("SIGBA", initializer=tf.constant(sbA))

params['UBB'] = tf.constant(ubB)
#params['UBB'] = tf.get_variable("UBB", initializer=tf.constant(ubB))

params['SIGBB'] = tf.constant(sbB)
#params['SIGBB'] = tf.get_variable("SIGBB", initializer=tf.constant(sbB))

params['PA'] =  tf.constant(pA)
#params['PA'] = tf.get_variable("PA", initializer=tf.constant(pA))

params['E'] = tf.constant(elj)
#params['E'] = tf.get_variable("E", initializer=tf.constant(elj))

params['UC'] = tf.constant(uc) #not optimize for now
#params['UC'] = tf.get_variable("UC", initializer=tf.constant(uc))

#params['NL'] =  tf.constant(nl)
params['NL'] =  tf.get_variable("NL", initializer=tf.constant(nl))

#params['PB'] =  tf.constant(pb)
params['PB'] = tf.get_variable("PB", initializer=tf.constant(pb))

#params['PC'] =  tf.constant(pc)
params['PC'] = tf.get_variable("PC", initializer=tf.constant(pc))



pi = tf.constant(np.pi)

ngauss_p = tf.constant(ngauss, dtype=tf.int32)
xgauss_p = tf.constant(np.float32(xgauss), dtype=tf.float32)
wgauss_p = tf.constant(np.float32(wgauss), dtype=tf.float32)

u_p = tf.constant(un, dtype=tf.float32)
lmbs_p = tf.constant(ln, dtype=tf.float32)

xasymp_p = tf.constant(xasymp, dtype=tf.float32)
dxasymp_p = tf.constant(dxasymp, dtype=tf.float32)
nx_p = tf.constant(nx, dtype=tf.int32)


pm_p = tf.constant(pm)

xc = tf.sqrt(params['UC']/params['E'])
a = tf.sqrt(1.+xc)
eps = xc/10.
xm = tf.sqrt(umax/params['E'])
nm = tf.pow(1. - params['NL']*a/xm, -1)

#Gaussian (mixture) for B for all input u's
gbA = tf.exp(-tf.pow(u_p-params['UBA'],2)/(2.0*tf.pow(params['SIGBA'],2)))/(tf.sqrt(2.*pi)*params['SIGBA'])
gbB = tf.exp(-tf.pow(u_p-params['UBB'],2)/(2.0*tf.pow(params['SIGBB'],2)))/(tf.sqrt(2.*pi)*params['SIGBB'])
gb1 = params['PA']*gbA + gbB - params['PA']*gbB

#gaussian at large distance pB(u - umax)
gbAfar = tf.exp(-tf.pow(u_p-umax-params['UBA'],2)/(2.0*tf.pow(params['SIGBA'],2)))/(tf.sqrt(2.*pi)*params['SIGBA'])
gbBfar = tf.exp(-tf.pow(u_p-umax-params['UBB'],2)/(2.0*tf.pow(params['SIGBB'],2)))/(tf.sqrt(2.*pi)*params['SIGBB'])
gbfar = params['PA']*gbAfar + gbBfar - params['PA']*gbBfar


#convolution of gaussian (mixture) and pwca
sq2 = tf.sqrt(2.)
#Pwca for all input y's
yA = sq2*params['SIGBA']*xgauss_p + u_p[:,None] - params['UBA']
x1A = tf.pow(yA/params['E'],2) #to make x positive
xA  = tf.pow( x1A , 0.25 )
bA = tf.sqrt(1.+ xA)
z1A = tf.tanh( tf.pow(a/bA,12.) )
zA = tf.pow( z1A, 1./12. ) #caps z to 1
fcore2A = tf.pow((1.-zA), params['NL']-1.)
fcore3A = a/((xA+eps)*tf.pow(bA,3)*4.*params['E'])
fcore4A = tf.sigmoid(20.*(yA-0.5*params['UC'])/params['UC'])
pwcaA = nm*params['NL']*fcore2A*fcore3A*fcore4A
#---
qA = tf.matmul(pwcaA,tf.reshape(wgauss_p,[ngauss_p,1]))/tf.sqrt(pi)
q2A = qA[:,0]

yB = sq2*params['SIGBB']*xgauss_p + u_p[:,None] - params['UBB']
x1B = tf.pow(yB/params['E'],2) #to make x positive
xB  = tf.pow( x1B , 0.25 )
bB = tf.sqrt(1.+ xB)
z1B = tf.tanh( tf.pow(a/bB,12.) )
zB = tf.pow( z1B, 1./12. ) #caps z to 1
fcore2B = tf.pow((1.-zB), params['NL']-1.)
fcore3B = a/((xB+eps)*tf.pow(bB,3)*4.*params['E'])
fcore4B = tf.sigmoid(20.*(yB-0.5*params['UC'])/params['UC'])
pwcaB = nm*params['NL']*fcore2B*fcore3B*fcore4B
#---
qB = tf.matmul(pwcaB,tf.reshape(wgauss_p,[ngauss_p,1]))/tf.sqrt(pi)
q2B = qB[:,0]

q2 = params['PA']*q2A + q2B - params['PA']*q2B

#p0's
p0 = params['PB']*gb1 + params['PC']*q2 + pm_p*gbfar

#kB1
klBA = tf.exp(0.5*tf.pow(params['SIGBA'],2)*tf.pow(lmbs_p,2) - lmbs_p*params['UBA'])
klBB = tf.exp(0.5*tf.pow(params['SIGBB'],2)*tf.pow(lmbs_p,2) - lmbs_p*params['UBB']) 
klB1 = params['PA']*klBA + klBB - params['PA']*klBB

#pwca for the x grid
x1_s = tf.pow(xasymp_p/params['E'],2) #to make x positive
x_s  = tf.pow( x1_s , 0.25 )
b_s = tf.sqrt(1.+ x_s)
z1_s = tf.tanh( tf.pow(a/b_s,12.) )
z_s = tf.pow( z1_s, 1./12. ) #caps z to 1
fcore2_s = tf.pow((1.-z_s), params['NL']-1.)
fcore3_s = a/((x_s+eps)*tf.pow(b_s,3)*4.*params['E'])
fcore4_s = tf.sigmoid(20.*(xasymp_p-0.5*params['UC'])/params['UC'])
pwca_s = nm*params['NL']*fcore2_s*fcore3_s*fcore4_s

#kwca
fsamples = dxasymp_p*pwca_s
expl = tf.exp(- xasymp_p * lmbs_p[:,None])
q_C = tf.matmul(expl,tf.reshape(fsamples,[nx_p,1]))
klwca = q_C[:,0]

#free energies
e2 = tf.exp(-umax * lmbs_p)
klC = params['PB'] + e2 - params['PC']*e2 - params['PB']*e2 + params['PC'] * klwca
kl = klB1*klC
pkl = p0/kl

#cost function
cost = -tf.reduce_sum(tf.log(pkl))
    
optimizer = tf.train.AdamOptimizer(2.e-6)
#optimizer = tf.train.GradientDescentOptimizer(5.e-6)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

gradient_cost = optimizer.compute_gradients(cost)

sess = tf.Session()

sess.run(init)

ll = sess.run(cost)
print("starting at x:", " uBA =", ubA*kT, "sbA =", sbA*kT, " uBB =", ubB*kT, "sbB =", sbB*kT, "pA =", pA, "pb = ", pb, "pc =", pc, "pm = ", pm, "umax = ", umax, "nl = ", nl, "eLJ = ", elj*kT, "uc = ", uc*kT)
print("cost:", ll)
print(sess.run(gradient_cost))

best_loss = ll
best_ubA = sess.run(params['UBA'])
best_sbA = sess.run(params['SIGBA'])
best_ubB = sess.run(params['UBB'])
best_sbB = sess.run(params['SIGBB'])
best_pA = sess.run(params['PA'])
best_pb = sess.run(params['PB'])
best_pc = sess.run(params['PC'])
best_nl = sess.run(params['NL'])
best_elj = sess.run(params['E'])
best_uc = sess.run(params['UC'])   
for step in range(100):
      sess.run(train)
      ll = sess.run(cost)
      lubA = sess.run(params['UBA'])
      lsbA = sess.run(params['SIGBA'])
      lubB = sess.run(params['UBB'])
      lsbB = sess.run(params['SIGBB'])
      lpA =  sess.run(params['PA'])
      lpb = sess.run(params['PB'])
      lpc = sess.run(params['PC'])
      lnl = sess.run(params['NL'])
      lelj = sess.run(params['E'])
      luc = sess.run(params['UC'])                     
      if( ll < best_loss ):
          best_loss = ll
          best_ubA = lubA
          best_sbA = lsbA
          best_ubB = lubB
          best_sbB = lsbB
          best_pA = lpA
          best_pb = lpb
          best_pc = lpc
          best_nl = lnl
          best_elj = lelj
          best_uc = luc
      print("step", step, "x:", " uBA =", lubA*kT, "sbA =", lsbA*kT, " uBB =", lubB*kT, "sbB =", lsbB*kT, "pA = ", lpA, "pb = ", lpb, "pc =", lpc, "pm = ", pm, "umax = ", umax, "nl = ", lnl, "eLJ = ", lelj*kT, "uc = ", luc*kT, "cost =", ll)
      print(sess.run(gradient_cost))

print("----- End of optimization --------");
print("best", "x:", " uBA =", best_ubA*kT, "sbA =", best_sbA*kT, "uBB =", best_ubB*kT, "sbB =", best_sbB*kT, "pA =", best_pA, "pb = ", best_pb, "pc =", best_pc, "pm = ", pm, "umax = ", umax, "nl = ", best_nl, "eLJ = ", best_elj*kT, "uc = ", best_uc*kT, "cost =", best_loss)
