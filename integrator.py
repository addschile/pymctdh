import numpy as np
from scipy.linalg import expm
from eom import eom,eom_coeffs,eom_spfs
#from eom import fulleom,eom_coeffs,eom_spfs
from optools import matel
from cy.wftools import norm,inner
from functools import lru_cache
import units as units

@lru_cache(maxsize=None,typed=False)
def get_butcher(method):
    """
    """
    a  = []
    b  = []
    c  = []
    e  = []
    bh = None
    if method == 'rk4':
        # a
        a.append( [] )
        a.append( [0./5.] )
        a.append( [0.0, 0.5] )
        a.append( [0.0, 0.0, 1.0] )
        # b
        b.append( [1./6., 1./3., 1./3., 1./6.] )
        # c
        c.append( 0.0 )
        c.append( 0.5 )
        c.append( 0.5 )
        c.append( 1.0 )
    elif method == 'rk5':
        # a
        a.append( [] )
        a.append( [1./5.] )
        a.append( [3./40., 9./40.] )
        a.append( [44./45., -56./15., 32./9.] )
        a.append( [19372./6561., -25360./2187., 64448./6561., -212./729.] )
        a.append( [9017./3168., -355./33., 46732./5247., 49./176., -5103./18656.] )
        a.append( [35./384., 0.0, 500./1113., 125./192. -2187./6784., 11./84.] )
        # b
        b.append( [35./384., 0.0, 500./1113., 125./192. -2187./6784., 11./84.] )
        # TODO double check these
        b.append( [5179./57600., 0.0, 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.] )
        # c
        c.append( 0.0 )
        c.append( 1./5. )
        c.append( 3./10. )
        c.append( 4./5. )
        c.append( 8./9. )
        c.append( 1.0 )
        c.append( 1.0 )
        # e
        e.append(  71.0/57600.0 )
        e.append( -71.0/16695.0 )
        e.append(  71.0/1920.0 )
        e.append( -17253.0/339200.0 )
        e.append(  22.0/525.0 )
        e.append( -1.0/40.0 )
    elif method == 'rk8':
        # a
        a.append( [] )
        a.append( [5.26001519587677318785587544488e-02] )
        a.append( [1.97250569845378994544595329183e-02, 5.91751709536136983633785987549e-02] )
        a.append( [2.95875854768068491816892993775e-02, 0.0, 8.87627564304205475450678981324e-02] )
        a.append( [2.41365134159266685502369798665e-01, 0.0, -8.84549479328286085344864962717e-01, 9.24834003261792003115737966543e-01] )
        a.append( [3.7037037037037037037037037037e-02, 0.0, 0.0, 1.70828608729473871279604482173e-01, 1.25467687566822425016691814123e-01] )
        a.append( [3.7109375e-02, 0.0, 0.0, 1.70252211019544039314978060272e-01, 6.02165389804559606850219397283e-02, -1.7578125e-02 ] )
        a.append( [3.70920001185047927108779319836e-02, 0.0, 0.0, 1.70383925712239993810214054705e-01, 1.07262030446373284651809199168e-01, -1.53194377486244017527936158236e-02, 8.27378916381402288758473766002e-03] )
        a.append( [6.24110958716075717114429577812e-01, 0.0, 0.0, -3.36089262944694129406857109825e+00, -8.68219346841726006818189891453e-01, 2.75920996994467083049415600797e+01, 2.01540675504778934086186788979e+01, -4.34898841810699588477366255144e+01] )
        a.append( [4.77662536438264365890433908527e-01, 0.0, 0.0, -2.48811461997166764192642586468e+00, -5.90290826836842996371446475743e-01, 2.12300514481811942347288949897e+01, 1.52792336328824235832596922938e+01, -3.32882109689848629194453265587e+01, -2.03312017085086261358222928593e-02] )
        a.append( [-9.3714243008598732571704021658e-01, 0.0, 0.0, 5.18637242884406370830023853209e+00, 1.09143734899672957818500254654e+00, -8.14978701074692612513997267357e+00, -1.85200656599969598641566180701e+01, 2.27394870993505042818970056734e+01, 2.49360555267965238987089396762e+00, -3.0467644718982195003823669022e+00] )
        a.append( [2.27331014751653820792359768449e+00, 0.0, 0.0, -1.05344954667372501984066689879e+01, -2.00087205822486249909675718444e+00, -1.79589318631187989172765950534e+01, 2.79488845294199600508499808837e+01, -2.85899827713502369474065508674e+00, -8.87285693353062954433549289258e+00, 1.23605671757943030647266201528e+01, 6.43392746015763530355970484046e-01] )
        # b
        b.append( [5.42937341165687622380535766363e-02, 0.0, 0.0, 0.0, 0.0, 4.45031289275240888144113950566e+00, 1.89151789931450038304281599044e+00, -5.8012039600105847814672114227e+00, 3.1116436695781989440891606237e-01, -1.52160949662516078556178806805e-01, 2.01365400804030348374776537501e-01, 4.47106157277725905176885569043e-02] )
        # bh
        bh = []
        bh.append( 0.244094488188976377952755905512e+00 )
        bh.append( 0.733846688281611857341361741547e+00 )
        bh.append( 0.220588235294117647058823529412e-01 )
        # c
        c.append( 0.0                               )
        c.append( 0.0526001519587677318785587544488 )
        c.append( 0.0789002279381515978178381316732 )
        c.append( 0.118350341907227396726757197510  )
        c.append( 0.281649658092772603273242802490  )
        c.append( 0.333333333333333333333333333333  )
        c.append( 0.25                              )
        c.append( 0.307692307692307692307692307692  )
        c.append( 0.651282051282051282051282051282  )
        c.append( 0.6                               )
        c.append( 0.857142857142857142857142857142  )
        # e
        e.append(  0.1312004499419488073250102996e-01 )
        e.append(  0.0 )
        e.append(  0.0 )
        e.append(  0.0 )
        e.append(  0.0 )
        e.append( -0.1225156446376204440720569753e+01 )
        e.append( -0.4957589496572501915214079952e+00 )
        e.append(  0.1664377182454986536961530415e+01 )
        e.append( -0.3503288487499736816886487290e+00 )
        e.append(  0.3341791187130174790297318841e+00 )
        e.append(  0.8192320648511571246570742613e-01 )
        e.append( -0.2235530786388629525884427845e-01 )

    if bh is None:
        return a,b,c,e
    else:
        return a,b,bh,c,e

def euler(t_start, t_finish, dt, y, ham, pbf, cmfflag=False, eq=None):
    """Runs forward euler integration for equations of motion. Only used for
    testing code really.
    """
    if not cmfflag:
        #k_A,k_spf = eom(t0,dt,nel,nmodes,nspf,npbf,spfstart,spfend,ham,pbf,y_A,y_spf)
        k_A,k_spf = eom(nel,nmodes,nspf,npbf,spfstart,spfend,ham,pbf,y_A,y_spf)
    y.A += dt*k_A
    y.spf += dt*k_spf

def rk(t_start, t_finish, dt, y, ham, pbfs, method='rk8', cmfflag=False, eq=None, 
       opspfs=None, opips=None, spfovs=None, mfs=None, rhos=None, projs=None):
    """Dormand-Prince Runge-Kutta based algorithms with adaptive time-stepping.
    """

    # get wavefunction info
    nel      = y.nel
    nmodes   = y.nmodes
    nspfs    = y.nspfs
    npbfs    = y.npbfs
    spfstart = y.spfstart
    spfend   = y.spfend

    # set initial time of step
    t0 = t_start

    # get runke-kutta info
    if method=='rk5':
        nsteps = 6
        order = 5
        a,b,c,e = get_butcher('rk5')
        safe   = 0.9
        beta   = 0.04
        facdec = 0.25
        facinc = 8.0
    elif method=='rk8':
        nsteps = 11
        order = 8
        a,b,bh,c,e = get_butcher('rk8')
        safe   = 0.9
        beta   = 0.04
        facdec = 0.10
        facinc = 6.0

    # step size control
    expo1  = 0.125-beta*0.2
    facold = safe**(1.0/(expo1-beta))
    facdc1 = 1.0/facdec
    facin1 = 1.0/facinc

    while abs(t0 - t_finish) > 1.e-12:
        # change timestep if too large
        if dt > (t_finish-t0):
            dt = t_finish-t0

        # do propagation
        if cmfflag:
            uopspfs = opspfs[0]
            copspfs = opspfs[1]
            uopips  = opips[0]
            copips  = opips[1]
            if eq == 'spfs':
                k_spfs = np.zeros(nsteps, np.ndarray)
                for i in range(nsteps):
                    y_spfs = y.copy(arg='spfs')
                    for j in range(len(a[i])):
                        if a[i][j] != 0.0:
                            y_spfs += k_spfs[j]*a[i][j]*dt
                    k_spfs[i] = eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                         uopspfs,copspfs,copips,ham.huelterms,
                                         ham.hcelterms,spfovs,y.A,y_spfs,
                                         mfs=mfs,rhos=rhos,projs=projs)
                    #k_spfs[i] = eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                    #                     ham,opspfs,opips,spfovs,y.A,y_spfs,
                    #                     mfs=mfs,rhos=rhos,projs=projs)
                # compute output
                for i in range(len(a[-1])):
                    if a[-1][i] != 0.0:
                        for j in range(nel):
                            y.spfs[j] += k_spfs[i][j]*a[-1][i]*dt
            elif eq == 'coeffs':
                k_A = np.zeros(nsteps, np.ndarray)
                for i in range(nsteps):
                    y_A = y.copy(arg='A')
                    for j in range(len(a[i])):
                        if a[i][j] != 0.0:
                            y_A += k_A[j]*a[i][j]*dt
                    k_A[i] = eom_coeffs(nel,nmodes,nspfs,npbfs,uopips,copips,
                                        ham.huelterms,ham.hcelterms,spfovs,y_A)
                    #k_A[i] = eom_coeffs(nel,nmodes,nspfs,npbfs,ham,opips,spfovs,y_A)
                # compute output
                for i in range(len(a[-1])):
                    if a[-1][i] != 0.0:
                        for j in range(nel):
                            y.A[j] += k_A[i][j]*a[-1][i]*dt
            else:
                raise ValueError('Not a valid eq')
        else:
            k_A = np.zeros(nsteps, np.ndarray)
            k_spfs = np.zeros(nsteps, np.ndarray)
            for i in range(nsteps):
                y_A = y.copy(arg='A')
                y_spfs = y.copy(arg='spfs')
                for j in range(len(a[i])):
                    if a[i][j] != 0.0:
                        y_A += k_A[j]*a[i][j]*dt
                        y_spfs += k_spfs[j]*a[i][j]*dt
                k_A[i],k_spfs[i] = eom(nel,nmodes,nspfs,npbfs,spfstart,
                                       spfend,ham,pbfs,y_A,y_spfs)
                #k_A[i],k_spfs[i] = fulleom(nel,nmodes,nspfs,npbfs,spfstart,
                #                       spfend,ham,pbfs,y_A,y_spfs)
            # compute output
            for i in range(len(a[-1])):
                if a[-1][i] != 0.0:
                    for j in range(nel):
                        y.A[j] += k_A[i][j]*a[-1][i]*dt
                        y.spfs[j] += k_spfs[i][j]*a[-1][i]*dt

        t0 += dt

#def rkadapt(t_start, t_finish, dt, y, dy_A, dy_spfs, ham, pbfs, method='rk8', cmfflag=False, eq=None, 
#       opspfs=None, opips=None, spfovs=None, mfs=None, rhos=None, projs=None):
#    """Dormand-Prince Runge-Kutta based algorithms with adaptive time-stepping.
#    """
#
#    # get wavefunction info
#    nel      = y.nel
#    nmodes   = y.nmodes
#    nspfs    = y.nspfs
#    npbfs    = y.npbfs
#    spfstart = y.spfstart
#    spfend   = y.spfend
#
#    # set initial time of step
#    t0 = t_start
#
#    # get runke-kutta info
#    if method=='rk5':
#        nsteps = 6
#        order = 5
#        a,b,c,e = get_butcher('rk5')
#        safe   = 0.9
#        beta   = 0.04
#        facdec = 0.25
#        facinc = 8.0
#        # step size control stuff
#        expo1  = 0.2-beta*0.75
#        facold = safe**(1.0/(expo1-beta))
#        facdc1 = 1.0/facdec
#        facin1 = 1.0/facinc
#    elif method=='rk8':
#        nsteps = 11
#        order = 8
#        a,b,bh,c,e = get_butcher('rk8')
#        safe   = 0.9
#        beta   = 0.04
#        facdec = 0.10
#        facinc = 6.0
#        # step size control
#        expo1  = 0.125-beta*0.2
#        facold = safe**(1.0/(expo1-beta))
#        facdc1 = 1.0/facdec
#        facin1 = 1.0/facinc
#
#    # initialize timestep guesser
#    dtnew = dt
#
#    # guess a timestep
#    if dt == 0.0:
#        dtnew = get_dt(t_finish-t0,y,dy_A,dy_spfs,ham,pbfs)
#
#    ### create useful arrays ###
#    # eom evaluation
#    k_A    = np.zeros(nsteps, dtype=np.ndarray)
#    k_spfs = np.zeros(nsteps, dtype=np.ndarray)
#    # error evaluation
#    k_Aerr    = np.zeros(nel, dtype=np.ndarray)
#    k_spfserr = np.zeros(nel, dtype=np.ndarray)
#
#    while abs(t0 - t_finish) > 1.e-12:
#        # set timestep
#        dt = dtnew
#
#        # change timestep if too large
#        if dt > (t_finish-t0):
#            dt = t_finish-t0
#
#        # evaluate runge-kutta steps
#        for i in range(1,nsteps+1):
#            y_A = y.copy(arg='A')
#            y_spfs = y.copy(arg='spfs')
#            for j in range(len(a[i])):
#                if a[i][j] != 0.0:
#                    y_A += k_A[j]*a[i][j]*dt
#                    y_spfs += k_spfs[j]*a[i][j]*dt
#            k_A[i],k_spfs[i] = eom(t0+c[i]*dt,dt,nel,nmodes,nspfs,npbfs,spfstart,
#                                   spfend,ham,pbfs,y_A,y_spfs)
#
#        # compute output change
#        for i in range(len(b)):
#            #if a[-1][i] != 0.0:
#            if b[i] != 0.0:
#                for j in range(nel):
#                    k_A[3][j]    += k_A[i][j]*b[i]*dt
#                    k_spfs[3][j] += k_spfs[i][j]*b[i]*dt
#                    #k_A[3][j]    += k_A[i][j]*a[-1][i]*dt
#                    #k_spfs[3][j] += k_spfs[i][j]*a[-1][i]*dt
#        for i in range(nel):
#            k_A[4][i]    = y.A[i] + k_A[3][i]
#            k_spfs[4][i] = y.spfs[i] + k_spfs[3][i]
#
#        # compute error change
#        for i in range(nel):
#            k_Aerr[i] = np.zeros_like(y.A[i], dtype=complex)
#            k_spfserr[i] = np.zeros_like(y.spfs[i], dtype=complex)
#            for j in range(len(e)):
#                if e[j] != 0.0:
#                    k_Aerr[i]    += k_A[i][j]*a[j]*dt
#                    k_spfserr[i] += k_spfs[i][j]*a[j]*dt
#
#        ### compute error ###
#        # sets weight of error for each element
#        wt_A    = np.zeros(nel, dtype=np.ndarray)
#        wt_spfs = np.zeros(nel, dtype=np.ndarray)
#        for i in range(nel):
#            wt_A[i]    = tol + tol*np.maximum(y.A[i],k[4][i])
#            wt_spfs[i] = tol + tol*np.maximum(y.spfs[i],k[4][i])
#        #CALL SETEWV(ndgl,y,k5,tol,uround,wt)
#        # computes weighted error
#        err = 0.0
#        for i in range(nel):
#            err += (np.abs(k_Aerr[i])/wt_A[i])**2.
#            err += (np.abs(k_spfserr[i])/wt_spfs[i])**2.
#        err = err
#        #err = wgtnrm(ndgl,kerr,wt)
#
#        k_Aerr2 = deepcopy(k_A[3]) #k4
#        k_Aerr2 += -bh1*k_A[0] #k1
#        k_Aerr2 += -bh2*k_A[8] #k9
#        k_Aerr2 += -bh3*k_A[2] #k3
#        k_spfserr2 = deepcopy(k_spfs[3]) #k4
#        k_spfserr2 += -bh1*k_spfs[0] #k1
#        k_spfserr2 += -bh2*k_spfs[8] #k9
#        k_spfserr2 += -bh3*k_spfs[2] #k3
#        #CALL zcopy(ndgl,     k4,1,kerr2,1)
#        #ztmp = DCMPLX(-bh1)
#        #CALL zaxpy(ndgl,ztmp,k1,1,kerr2,1)
#        #ztmp = DCMPLX(-bh2)
#        #CALL zaxpy(ndgl,ztmp,k9,1,kerr2,1)
#        #ztmp = DCMPLX(-bh3)
#        #CALL zaxpy(ndgl,ztmp,k3,1,kerr2,1)
#
#        err2 = 0.0
#        for i in range(nel):
#            err2 += (np.abs(k_Aerr2[i])/wt_A[i])**2.
#            err2 += (np.abs(k_spfserr2[i])/wt_spfs[i])**2.
#        #err2 = wgtnrm(ndgl,kerr2,wt)
#        deno = err + 0.01*err2
#        if deno <= 0.0: deno = 1.0
#        err = dt*(err/np.sqrt(deno))
#
#        # Computation of new timestep
#        fac1 = err**expo1
#        # LUND-stabilization
#        fac = fac1/facold**beta
#        # We require  facdec <= hnew/h <= facinc
#        fac   = max(facin1,MIN(facdc1,fac/safe))
#        dtnew = dt/fac
#
#        if err < 1.0:
#            # step is accepted
#            t0 += dt
#            # update wavefunction
#            y.A = deepcopy(k_A[4])
#            y.spfs = deepcopy(k_spfs[4])
#            # New model evaluation
#            dy_A,dy_spfs = eom(t0,dt,nel,nmodes,nspfs,npbfs,spfstart,
#                               spfend,ham,pbfs,y_A,y_spfs)
#            #CALL ODEMOD(t,k5,k4,mc,mr,mi,ml)
#            facold = MAX(err,1.0d-4)
#        else:
#            dtnew   = dt/min(facdc1,fac1/safe)
#            # TODO not sure about this
#            dy_A    = deepcopy(k_A[3])
#            dy_spfs = deepcopy(k_spfs[3])
#    #return energy,error
#
#def get_dt():
#    """
#    """
#
#    # precompute stuff
#    # sets weight of error for each element
#    wt_A    = np.zeros(nel, dtype=np.ndarray)
#    wt_spfs = np.zeros(nel, dtype=np.ndarray)
#    for i in range(nel):
#        wt_A[i]    = tol + tol*y.A[i]
#        wt_spfs[i] = tol + tol*y.spfs[i]
#    #call SETEWV(ndgl,y,y,tol,small,wt)
#    ynrm = 0.0
#    dynrm = 0.0
#    for i in range(nel):
#        ynrm += (np.abs(y.A[i])/wt_A[i])**2.
#        ynrm += (np.abs(y.spfs[i])/wt_spfs[i])**2.
#        dynrm += (np.abs(dy_A[i])/wt_A[i])**2.
#        dynrm += (np.abs(dy_spfs[i])/wt_spfs[i])**2.
#    #ynrm  = wgtnrm(ndgl, y, wt)
#    #dynrm = wgtnrm(ndgl, dy0, wt)
#    # COMPUTE STEPSIZE FOR EXPLICIT EULER SUCH THAT
#    # THE INCREMENT IS SMALL COMPARED TO THE SOLUTION
#    if ((dynrm <= small) or (ynrm <= small)):
#       dt = 1.0d-6
#    else:
#       dt = max(dtmax*small,abs((ynrm/dynrm)*0.01))
#    dtmx = min(abs(tend-t),dtmax)
#    dt   = min(dt,dtmx)
#    # PERFORM AN EXPLICIT EULER STEP
#    CALL zcopy(ndgl,y,1,y1,1)
#    ztmp = DCMPLX(h)
#    CALL zaxpy(ndgl,ztmp,dy0,1,y1,1)
#    # EVALUATE
#    CALL ODEMOD(t+h,y1,dy1,mc,mr,mi,ml)
#    # ESTIMATE THE SECOND DERIVATIVE OF THE SOLUTION
#    wt_A    = np.zeros(nel, dtype=np.ndarray)
#    wt_spfs = np.zeros(nel, dtype=np.ndarray)
#    for i in range(nel):
#        wt_A[i]    = tol + tol*np.maximum(y.A[i],y1_A)
#        wt_spfs[i] = tol + tol*np.maximum(y.spfs[i],y1_A)
#    #CALL SETEWV(ndgl,y,y1,tol,small,wt)
#    ynrm = 0.0
#    dynrm = 0.0
#    for i in range(nel):
#        ynrm += (np.abs(y.A[i])/wt_A[i])**2.
#        ynrm += (np.abs(y.spfs[i])/wt_spfs[i])**2.
#        dynrm += (np.abs(dy_A[i])/wt_A[i])**2.
#        dynrm += (np.abs(dy_spfs[i])/wt_spfs[i])**2.
#    ddynrm = 0.0
#    for i in range(nel):
#        ddynrm += (abs(dy_A[i]-dy1_A[i])/wt_A[i])**2.
#    # TODO
#    ddynrm = np.sqrt(ddynrm)
#    wgtdst = SQRT(tmp/ndim)
#    #ddynrm = wgtdst(ndgl,dy0,dy1,wt)
#    ddynrm = ddynrm/dt
#    # COMPUTE INITIAL STEP SIZE
#    dnrmmx = max(abs(ddynrm),dynrm)
#    if (dnrmmx <= 1.e-15):
#       dttry = max(1.0e-6,abs(dt)*1.0e-3)
#    else:
#       dttry = (0.010/dnrmmx)**(1.0/iord)
#    dtnew = min(1.0e2*dt,dttry,dtmx)
#    return dtnew
#
## TODO
#def SETEWV(ndim, y0, y1, tol, small, wt):
#    """
#    """
#    integer    ndim, i
#    real*8     tol, small, wt(ndim)
#    complex*16 y0(ndim), y1(ndim)
#    do i = 1,ndim
#       wt(i) = tol + tol*max(abs(y0(i)), abs(y1(i)), small)
#    enddo
#    return
#
#def wgtnrm(ndim, vec, wt):
#    """
#    """
#INTEGER    i,ndim
#REAL*8     tmp,wt(ndim)
#COMPLEX*16 vec(ndim)
#    tmp = 0.d0
#    DO i = 1,ndim
#       tmp = tmp + (abs(vec(i))/wt(i))**2
#    ENDDO
#    wgtnrm = np.sqrt(tmp/ndim)
#    return

################################################################################
# Generalized krylov subspace method functions                                 #
################################################################################
def lanczos(y, ham, uopips, copips, spfovs, nvecs=5, v0=None, return_evecs=True):
    """
    """
    # get wavefunction info
    nel      = y.nel
    nmodes   = y.nmodes
    nspfs    = y.nspfs
    npbfs    = y.npbfs
    spfstart = y.spfstart
    spfend   = y.spfend
    A        = y.A

    # thing to store A tensors
    V = np.zeros(nvecs, np.ndarray)
    V[0] = A/norm(nel,A)
    T = np.zeros((nvecs,nvecs))

    # form krylov vectors and tridiagonal matrix
    for i in range(nvecs-1):
        V[i+1] = matel(nel,nmodes,nspfs,npbfs,uopips,copips,ham.huelterms,
                       ham.hcelterms,spfovs,V[i])
        # compute alpha 
        T[i,i] = inner(nel,V[i],V[i+1]).real
        if i>0:
            V[i+1] += -T[i,i]*V[i] - T[i-1,i]*V[i-1]
        else:
            V[i+1] += -T[i,i]*V[i]
        # normalize previous vector
        nvprev = norm(nel,V[i+1])
        V[i+1] /= nvprev
        # compute beta
        T[i,i+1] = nvprev
        T[i+1,i] = T[i,i+1]
    T[-1,-1] = inner(nel,V[-1], matel(nel,nmodes,nspfs,npbfs,uopips,copips,
                                      ham.huelterms,ham.hcelterms,spfovs,V[-1])).real

    if return_evecs:
        return T , V
    else:
        return T

def arnoldi(y, ham, uopips, copips, spfovs, nvecs=5, v0=None, return_evecs=True):
    """
    """
    # get wavefunction info
    nel      = y.nel
    nmodes   = y.nmodes
    nspfs    = y.nspfs
    npbfs    = y.npbfs
    spfstart = y.spfstart
    spfend   = y.spfend
    A        = y.A

    # thing to store A tensors
    V = np.zeros(nvecs, np.ndarray)
    V[0] = A/norm(nel,A)

    # form krylov subspace and upper hessenberg matrix
    T = np.zeros((nvecs,nvecs), dtype=complex)
    for j in range(nvecs-1):
        w = matel(nel,nmodes,nspfs,npbfs,uopips,copips,ham.huelterms,
                  ham.hcelterms,spfovs,V[j])
        for i in range(j+1):
            T[i,j] = inner(nel, w, V[i])
            w -= T[i,j]*V[i]
        if j < nvecs-1:
            T[j+1,j] = norm(nel,w)
            V[j+1] = w/T[j+1,j]

    return T , V

################################################################################
# Functions that use krylov subspace methods for oring all the vectors         #
################################################################################
def propagate(V, T, dt):
    """
    """
    nvecs = len(V)
    psiprop = expm(-1.j*dt*T)[:,0]
    for i in range(nvecs):
        if i==0:
            psiout = psiprop[i]*V[i]
        else:
            psiout += psiprop[i]*V[i]
    return psiout

def krylov_prop(t_start, t_finish, dt, y, ham, uopips, copips, spfovs,
                method='lanczos', return_all=False):
    if method == 'arnoldi':
        T , V = arnoldi(y, ham, uopips, copips, spfovs, nvecs=5)
    else:
        T , V = lanczos(y, ham, uopips, copips, spfovs, nvecs=20)
    y.A = propagate(V, T, dt)
    #psiout = propagate(V, T, dt)
    #if return_all:
    #    return psiout , T , V
    #else:
    #    return psiout
