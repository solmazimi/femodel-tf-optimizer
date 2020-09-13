from __future__ import print_function
import pandas as pd
import numpy as np
import random
import math
import pickle
from numpy.polynomial.hermite import hermgauss
import tensorflow as tf
from femodel_tf_optimizer import femodel_tf_optimizer
import os, sys
import argparse

class femodel(femodel_tf_optimizer):
    """
    Class to optimize analytic binding free energy model with Tensorflow
    """
    # rational soft-core
    kBT = 1.986e-3*300.0
    alphasc = tf.constant(1./16.,dtype=tf.float64)
    umaxsc = tf.constant(50.0/kBT, dtype=tf.float64)
    ausc = alphasc*umaxsc
    ubcore = tf.constant(0.0/kBT, dtype=tf.float64)
    def inverse_soft_core_function(self,usc):
        return tf.where(usc <= femodel.ubcore,
                        usc,
                        femodel.ubcore + 0.5*femodel.ausc*(-1.0 + tf.math.sqrt(2.*tf.math.pow((femodel.umaxsc+(usc-femodel.ubcore))/(femodel.umaxsc-(usc-femodel.ubcore)),1./femodel.alphasc)-1.0)))
    
    def der_soft_core_function(self,u):# dusc/du
        self.yscaX = tf.math.pow(1. + 2.*(u-femodel.ubcore)/femodel.ausc + 2.*tf.math.pow((u-femodel.ubcore)/femodel.ausc,2), femodel.alphasc)
        return tf.where(u <= femodel.ubcore,
                        tf.ones([tf.size(u)], dtype=tf.float64),
                        4.*femodel.ausc*(femodel.ausc + 2.*(u-femodel.ubcore))*self.yscaX/\
                        ((2.*tf.math.pow(femodel.ubcore,2) + tf.math.pow(femodel.ausc,2) + 2.*femodel.ausc*u+2.*u*u-2.*femodel.ubcore*(femodel.ausc+2.*u))*tf.math.pow(1.+self.yscaX,2)) )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optimize", action="store_true", default=0,
                        help="Optimize parameters. Otherwise plot only.")
    parser.add_argument("-c", "--ncycles", type=int, default=100,
                        help="Number of optimization cycles")
    parser.add_argument("-n", "--nsteps", type=int, default=10,
                        help="Number of optimization steps in each cycle")
    parser.add_argument("-r", "--restart", action="store_true", default=0,
                        help="Restart optimization from the parameters saved in the pickle restart file")
    parser.add_argument("-b", "--basename",
                        help="Basename of the optimization job. Defaults to basename of this script.")
    parser.add_argument("-d", "--datafile", default="repl.cycle.potE.temp.lambda.ebind.lambda1.lambda2.alpha.u0.w0.dat",
                        help="Data file")
    
    parser.add_argument("-l", "--plambda", type=float, default=0.0,
                        help="Value of lambda at which to plot the distributions")
    args = parser.parse_args()
    
    restart = args.restart
    production = args.optimize
    testarea = not production

    if args.basename:
        basename = args.basename
    else:
        basename = os.path.basename(os.path.splitext(sys.argv[0])[0])
        
    datafile = args.datafile
    ncycles = args.ncycles
    nsteps = args.nsteps
    tlambda = args.plambda
    
    sdm_data_raw = pd.read_csv(datafile, delim_whitespace=True, 
                           header=None,names=["replica","cycle","potE",
                                              "temp","Lambda","ebind",
                                              "Lambda1", "Lambda2", "alpha", "u0", "w0" ])
#    sdm_data = sdm_data_raw[sdm_data_raw['Lambda'] > 0.075 ]
#    sdm_data = sdm_data_raw[sdm_data_raw['cycle'] % 4 == 0 ]
    sdm_data = sdm_data_raw
    
    temperature = 300.0
    kT = 1.986e-3*temperature # [kcal/mol]
    beta = 1./kT

    nmodes = 3

    reference_params = {}        
    reference_params['ub'] = [  -15.35*beta, -10.7*beta, 13.7*beta ]
    reference_params['sb'] = [   3.24*beta, 3.22*beta, 3.0*beta ]
    reference_params['pb'] = [   9.31e-8,    6.13e-8, 1.e-20 ]
    reference_params['elj'] = [ 4.93*beta, 7.91*beta, 18.5*beta ]
    reference_params['uce'] = [ 2.52/4.93, 15.1/7.91, 72.0/18.5 ]                 
    reference_params['nl']  = [ 4.87, 6.10, 17.5 ]
    reference_params['wg'] = [  1.94e-8, 2.31e-7, 1.0  ]
    
    scale_params = {}
    scale_params['ub'] =  [ 1.* beta, 1.* beta, 1.* beta]
    scale_params['sb'] =  [ 0.1*beta, 0.1*beta, 0.1*beta]
    scale_params['pb'] =  [ 1.e-8, 1.e-8, 1.e-20]
    scale_params['elj'] = [ 1.*beta, 1.*beta,  1.*beta]
    scale_params['uce'] = [ 0.1, 0.1, 0.1 ]
    scale_params['nl']  = [ 1.0, 1.0, 1.0 ]
    scale_params['wg'] =  [ 1.0e-8, 1.0e-7, 1.0  ]


    learning_rate = 0.01

    discard = 0
    
    xparams = {}
    if restart:
        with open(basename + '.pickle', 'rb') as f:
            best_ubx, best_sbx, best_pbx, best_ex, best_ucx, best_nlx, best_wgx = pickle.load(f)
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

    fe_optimizer = femodel(sdm_data, reference_params, temperature,
                  xparams=xparams, scale_params=scale_params, discard=discard, learning_rate=learning_rate)
                                        
    #variables = [ fe_optimizer.pbx_t, fe_optimizer.ex_t, fe_optimizer.ucx_t, fe_optimizer.nlx_t, fe_optimizer.wgx_t ]
    variables = [ fe_optimizer.ubx_t, fe_optimizer.sbx_t , fe_optimizer.wgx_t ]
    
    #----- test area ----------------

    tf.logging.set_verbosity(tf.logging.ERROR)
    
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

        # LAMBDAS= ' 0.000, 0.067, 0.133, 0.200, 0.267, 0.333, 0.400, 0.467, 0.533, 0.600, 0.667, 0.733, 0.800, 0.867, 0.933, 1.000'
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
        
            for step in range(ncycles):
                for i in range(nsteps):
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

                with open(basename + '.pickle', 'wb') as f:
                    pickle.dump([best_ubx, best_sbx, best_pbx, best_ex, best_ucx, best_nlx, best_wgx],f)


                
            print("----- End of optimization --------");
            print("best", "cost =", best_loss)
            print(best_wg)
            print(fe_optimizer.applyunits(best_ub, best_sb, best_pb, best_elj, best_uce, best_nl))


