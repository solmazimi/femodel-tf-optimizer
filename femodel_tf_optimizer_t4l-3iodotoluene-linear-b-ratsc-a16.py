from __future__ import print_function
import pandas as pd
import numpy as np
import random
import math
import pickle
from numpy.polynomial.hermite import hermgauss
import tensorflow as tf
from femodel_tf_optimizer import femodel_tf_optimizer


restart = False
production = True
testarea = not production


class femodel_tf_optimizer_t4l(femodel_tf_optimizer):
    """
    Class to optimize analytic binding free energy model with Tensorflow
    """
    # rational soft-core
    alphasc = tf.constant(1./16.,dtype=tf.float64)
    umaxsc = 50.0/(1.986e-3*300.0)
    ausc = alphasc*umaxsc
    def inverse_soft_core_function(self,usc):
        return tf.where(usc <= 0.,
                        usc,
                        0.5*femodel_tf_optimizer_t4l.ausc*(-1.0 + tf.math.sqrt(2.*tf.math.pow((femodel_tf_optimizer_t4l.umaxsc+usc)/(femodel_tf_optimizer_t4l.umaxsc-usc),1./femodel_tf_optimizer_t4l.alphasc)-1.0)))
    
    def der_soft_core_function(self,u):# dusc/du
        self.yscaX = tf.math.pow(1. + 2.*u/femodel_tf_optimizer_t4l.ausc + 2.*tf.math.pow(u/femodel_tf_optimizer_t4l.ausc,2), femodel_tf_optimizer_t4l.alphasc)
        return tf.where(u <= 0.,
                        tf.ones([tf.size(u)], dtype=tf.float64),
                        4.*femodel_tf_optimizer_t4l.ausc*(femodel_tf_optimizer_t4l.ausc + 2.*u)*self.yscaX/((femodel_tf_optimizer_t4l.ausc*femodel_tf_optimizer_t4l.ausc + 2.*femodel_tf_optimizer_t4l.ausc*u+2.*u*u)*tf.math.pow(1.+self.yscaX,2)))

if __name__ == '__main__':

    basename = "t4l-3iodotoluene-linear-b-ratsc-a16"
    datafile = "data/" + basename + "/repl.cycle.totE.potE.temp.lambda.ebind.lambda1.lambda2.alpha.u0.w0.dat"
    sdm_data = pd.read_csv(datafile, delim_whitespace=True, 
                           header=None,names=["replica","cycle","totE",
                                              "potE","temp","Lambda","ebind",
                                              "Lambda1", "Lambda2", "alpha", "u0", "w0" ])
    temperature = 300.0
    kT = 1.986e-3*temperature # [kcal/mol]
    beta = 1./kT

    nmodes = 3
    
    reference_params = {}        
    reference_params['ub'] = [  -11.0*beta, -4.00*beta, 100.0*beta ]
    reference_params['sb'] = [   1.95*beta, 2.80*beta, 10.*beta ]
    reference_params['pb'] = [ 1.e-5, 1.e-6, 0. ]
    reference_params['elj'] = [ 20.*beta, 20.*beta, 20.*beta ]
    reference_params['uce'] = [ 0., 0., 0. ]                 
    reference_params['nl']  = [ 5.5, 5.5, 8.0 ]
    reference_params['wg'] = [ 7.5e-3, 0.1, 1.0 ]
    
    scale_params = {}
    scale_params['ub'] =  [ 1.* beta, 1.* beta, 1.*beta]
    scale_params['sb'] =  [ 0.1*beta, 0.1*beta, 0.1*beta]
    scale_params['pb'] =  [ 1.e-5, 1.e-6, 1.e-6]
    scale_params['elj'] = [ 1.*beta, 1.*beta, 1.*beta ]
    scale_params['uce'] = [ 0.1, 0.1, 0.1 ]
    scale_params['nl']  = [ 0.1, 0.1, 0.1 ]
    scale_params['wg'] =  [ 1.e-3, 1.e-2, 1.0 ]

    learning_rate = 0.01

    discard = 0
    
    xparams = {}
    if restart:
        with open(basename + '.pickle', 'rb') as f:
            best_ubx, best_sbx, best_pbx, best_ex, best_ucx, best_nlx, best_weight = pickle.load(f)
            xparams['ub'] = best_ubx
            xparams['sb'] = best_sbx
            xparams['pb']  = best_pbx
            xparams['elj']   = best_ex 
            xparams['uce']  = best_ucx
            xparams['nl']  = best_nlx
            xparams['wg'] = best_wgx
    else:
        xparams['ub']  = [0. for i in range(nmodes) ]
        xparams['sb']  = [0. for i in range(nmodes) ]
        xparams['pb']  = [0. for i in range(nmodes) ]
        xparams['elj']   = [0. for i in range(nmodes) ]
        xparams['uce']  = [0. for i in range(nmodes) ]
        xparams['nl']  = [0. for i in range(nmodes) ]
        xparams['wg']  = [0. for i in range(nmodes) ]

    fe_optimizer = femodel_tf_optimizer_t4l(sdm_data, reference_params, temperature,
                                        xparams=xparams, scale_params=scale_params, discard=discard, learning_rate=learning_rate)
                                        
    variables = [ fe_optimizer.pbx_t, fe_optimizer.ex_t, fe_optimizer.ucx_t, fe_optimizer.nlx_t, fe_optimizer.wgx_t ]
    
    #----- test area ----------------


    
    if testarea:
        with tf.Session() as sess:
            sess.run(fe_optimizer.init)
            ll = sess.run(fe_optimizer.cost)
            uv = sess.run(fe_optimizer.u)
            uscv = sess.run(fe_optimizer.usc)
            p0v = sess.run(tf.squeeze(fe_optimizer.p0))
            p0scv = sess.run(tf.squeeze(fe_optimizer.p0sc))
            pklv = sess.run(tf.squeeze(fe_optimizer.pkl))
            xscv = sess.run(fe_optimizer.xsc)
            p0scxv = sess.run(tf.squeeze(fe_optimizer.p0scx))
            lv = sess.run(fe_optimizer.lambdas['Lambda'])
            upscv = sess.run(fe_optimizer.upsc)
            klv = sess.run(tf.squeeze(fe_optimizer.kl))


            best_ubx = sess.run(fe_optimizer.ubx_t)
            best_sbx = sess.run(fe_optimizer.sbx_t)
            best_pbx = sess.run(fe_optimizer.pbx_t)
            best_ex  = sess.run(fe_optimizer.ex_t)
            best_ucx = sess.run(fe_optimizer.ucx_t)
            best_nlx = sess.run(fe_optimizer.nlx_t)
            best_wgx = sess.run(fe_optimizer.wgx_t)
                        
            best_ub = sess.run(fe_optimizer.ub_t)
            best_sb = sess.run(fe_optimizer.sb_t)
            best_pb = sess.run(fe_optimizer.pb_t)
            best_elj  = sess.run(fe_optimizer.elj_t)
            best_uce = sess.run(fe_optimizer.uce_t)
            best_nl = sess.run(fe_optimizer.nl_t)
            best_wg = sess.run(fe_optimizer.wg_t)
            
            print("start ", "cost =", ll)
            print("parameters:")
            print(best_wg)
            print(fe_optimizer.applyunits(best_ub, best_sb, best_pb, best_elj, best_uce, best_nl)) 
            print("-----")

        #LAMBDAS = '0.000, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0'
        tlambda = 0.229
        mask = abs(lv - tlambda) < 1.e-6
        hist, bin_edges = np.histogram(uscv[mask]*kT, bins=30, density=True)
        np = len(hist)
        dx = bin_edges[1] - bin_edges[0]
        xp = bin_edges[0:np] + 0.5*dx
        
        import matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(xp, hist, 'o', markersize = 1)
        ax.plot(uscv[mask]*kT, pklv[mask]/kT, '+', markersize = 1)
        #ax.set_xlim([-40*kT,200*kT])
        plt.show()




    #------- training area ----------------------

    if production:
        with tf.Session() as sess:
    
            sess.run(fe_optimizer.init)
            fe_optimizer.opt = fe_optimizer.optimizer.minimize(fe_optimizer.cost, var_list = variables)
            sess.graph.finalize()
            ll = sess.run(fe_optimizer.cost)

            best_loss = ll
            best_ubx = sess.run(fe_optimizer.ubx_t)
            best_sbx = sess.run(fe_optimizer.sbx_t)
            best_pbx = sess.run(fe_optimizer.pbx_t)
            best_ex  = sess.run(fe_optimizer.ex_t)
            best_ucx = sess.run(fe_optimizer.ucx_t)
            best_nlx = sess.run(fe_optimizer.nlx_t)
            best_wgx = sess.run(fe_optimizer.wgx_t)
            
            best_ub = sess.run(fe_optimizer.ub_t)
            best_sb = sess.run(fe_optimizer.sb_t)
            best_pb = sess.run(fe_optimizer.pb_t)
            best_elj  = sess.run(fe_optimizer.elj_t)
            best_uce = sess.run(fe_optimizer.uce_t)
            best_nl = sess.run(fe_optimizer.nl_t)
            best_wg = sess.run(fe_optimizer.wg_t)
            
            gubx = sess.run(fe_optimizer.gubx_t)
            gsbx = sess.run(fe_optimizer.gsbx_t)
            gpbx = sess.run(fe_optimizer.gpbx_t)
            gex = sess.run(fe_optimizer.gex_t)
            gucx = sess.run(fe_optimizer.gucx_t)
            gnlx = sess.run(fe_optimizer.gnlx_t)
            gwgx = sess.run(fe_optimizer.gwgx_t)
            print("gradients:")
            print(gubx,gsbx,gpbx,gex,gucx,gnlx,gwgx)
            
            print("start ", "cost =", ll)
            print("parameters:")
            print(best_wg)
            print(fe_optimizer.applyunits(best_ub, best_sb, best_pb, best_elj, best_uce, best_nl)) 
            print("-----")
        
            for step in range(10):
                for i in range(10):
                    sess.run(fe_optimizer.train) #all variables optimized
                    #sess.run(fe_optimizer.opt) #only selected variables are optimized
                gubx = sess.run(fe_optimizer.gubx_t)
                gsbx = sess.run(fe_optimizer.gsbx_t)
                gpbx = sess.run(fe_optimizer.gpbx_t)
                gex = sess.run(fe_optimizer.gex_t)
                gucx = sess.run(fe_optimizer.gucx_t)
                gnlx = sess.run(fe_optimizer.gnlx_t)
                gwgx = sess.run(fe_optimizer.gwgx_t)
                notok = ( np.any(np.isnan(gubx)) or
                          np.any(np.isnan(gsbx)) or
                          np.any(np.isnan(gpbx)) or
                          np.any(np.isnan(gex))  or
                          np.any(np.isnan(gucx)) or
                          np.any(np.isnan(gnlx)) or
                          np.any(np.isnan(gwgx)) )
                print("gradients:")
                print(gubx,gsbx,gpbx,gex,gucx,gnlx,gwgx)
                if notok:
                    print("Gradient error")
                    break
                ll = sess.run(fe_optimizer.cost)

                l_ubx = sess.run(fe_optimizer.ubx_t)
                l_sbx = sess.run(fe_optimizer.sbx_t)
                l_pbx = sess.run(fe_optimizer.pbx_t)
                l_ex  = sess.run(fe_optimizer.ex_t)
                l_ucx = sess.run(fe_optimizer.ucx_t)
                l_nlx = sess.run(fe_optimizer.nlx_t)
                l_wgx = sess.run(fe_optimizer.wgx_t)
            
                l_ub = sess.run(fe_optimizer.ub_t)
                l_sb = sess.run(fe_optimizer.sb_t)
                l_pb = sess.run(fe_optimizer.pb_t)
                l_elj = sess.run(fe_optimizer.elj_t)
                l_uce = sess.run(fe_optimizer.uce_t)
                l_nl = sess.run(fe_optimizer.nl_t)
                l_wg = sess.run(fe_optimizer.wg_t)
                
                if( ll < best_loss ):
                    best_loss = ll
                
                    best_ubx = l_ubx 
                    best_sbx = l_sbx
                    best_pbx = l_pbx
                    best_ex  = l_ex 
                    best_ucx = l_ucx
                    best_nlx = l_nlx 
                    best_wgx = l_wgx

                    best_ub  = l_ub  
                    best_sb  = l_sb 
                    best_pb  = l_pb 
                    best_elj = l_elj
                    best_uce = l_uce
                    best_nl  = l_nl   
                    best_wg  = l_wg
    
                print(step, "cost =", ll)
                print("parameters:")
                print(l_wg)
                print(fe_optimizer.applyunits(l_ub, l_sb, l_pb, l_elj, l_uce, l_nl)) 
                print("-----")
            
            print("----- End of optimization --------");
            print("best", "cost =", best_loss)
            print(best_wg)
            print(fe_optimizer.applyunits(best_ub, best_sb, best_pb, best_elj, best_uce, best_nl))

        with open(basename + '.pickle', 'wb') as f:
            pickle.dump([best_ubx, best_sbx, best_pbx, best_ex, best_ucx, best_nlx, best_wgx],f)

