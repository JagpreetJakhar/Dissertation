import numpy as np

class DW:
    def __init__ (self, Ms, A, alpha, L, a, b, hconst, f, htime = None):
        self.mu_0 = np.pi*4e-7
        self.Ms = Ms
        self.A = A
        self.alpha = alpha
        self.L = L
        self.a = a
        self.b = b
        self.hconst = hconst
        self.Bconst = self.hconst * self.mu_0
        self.f = f
        self.omega = 2*np.pi*self.f

        self.htime = htime

        # cross section
        self.S = L[1]*L[2]

        r1 = L[1]/L[2]
        r2 = L[2]/L[1]
        Ny = 0.0884731  #1 - (2*np.arctan(r1) + 0.5*r2*np.log10(1 + r1**2) - 0.5*r1*np.log10(1 + r2**2))/np.pi
        Nz = 0.911527 #1 - (2*np.arctan(r2) - 0.5*r2*np.log10(1 + r1**2) + 0.5*r1*np.log10(1 + r2**2))/np.pi
        self.Bk = self.mu_0*Ms*(Nz - Ny)

        self.X = 0.0
        self.phi = 0.0

        self.delta = self.DW_width(self.phi)

        self.ap = - 1e-9*self.a /(self.Ms*self.S)
        self.bp = -2*(1e-27)*self.b /(self.Ms*self.S)
        self.gamma = 1.76e2 # in per nanosecond per Tesla
        self.beta = self.gamma/(1+self.alpha**2)

    def DW_width(self, phi):
        dw = 1e9*np.pi*np.sqrt( 2*self.A / (self.mu_0*self.Ms**2*np.sin(phi)**2 + self.Ms*self.Bk ))
        return dw

    def fields(self, x, phi, t):
        Bx = self.Bapp(t) + self.ap * x + self.bp * x**3
        Bphi = -0.5*self.Bk*np.sin(2*phi)
        return Bx, Bphi

    def Bapp(self, t):
        if self.htime is not None:
            return (self.Bconst + self.mu_0 * self.htime(t))* np.sin(self.omega*t)
        else:
            return self.Bconst * np.sin(self.omega*t)
        
    def Happ(self, t):
        return self.Bapp(t)/self.mu_0

    def Epin(self, X):
        return self.a * X**2 + self.b*X**4

    def xfield(self, x, t):
        return self.Bapp(t) + self.ap * x + self.bp * x**3

    def set_hconst(self, hconst):
        self.hconst = hconst
        self.Bconst = self.hconst * self.mu_0
        return



def DW_EoM(t, y, params):
    """
    DW oscillator equation of motion

    Parameters
    ----------
    t  :  time
    y  :  array containing the DW position and angle
    params  :  DW class object containing all the parameter functions

    Returns
    ----------
    gradient : array of equation of motion
    """
    x, phi = y
    Bx, Bphi = params.fields(x, phi, t)
    dx = params.DW_width(phi)*params.beta*(params.alpha * Bx - Bphi)
    dphi = params.beta*(params.alpha*Bphi + Bx)
    return [dx, dphi]




class field_sequence:
    """
    callable class to produce a time dependent field sequence
    """
    def __init__ (self, fields, periods):
        import numpy as np

        self.fields = fields
        self.periods = periods
        self.periods_sum = np.cumsum(periods)

    def __call__ (self, t):
        if t < 0.0:
            val = 0.0
        elif t >= self.periods_sum[-1]:
            val = 0.0
        else:
            t_diff = self.periods_sum - t 
            n = 0
            for i in range(len(t_diff)):
                if t_diff[i] >= 0.0:
                    n = i 
                    break
            val = self.fields[n]
        return val


def run_field_sequence(field_low = 0.0, field_high = 1000.0, N_fields = 10, T = 4, dt = 0.1, y0 = [0.0, 0.0]):
    from scipy.integrate import solve_ivp
    rng = np.random.default_rng()
    fields = rng.uniform(field_low, field_high, N_fields)
    periods = np.ones(len(fields))*T
    total_time = np.sum(periods)
    print(fields)
    print(periods)

    htime = field_sequence(fields, periods)

    dw1 = DW(477e3, 1.05e-11, 0.02, (600e-9, 50e-9, 5e-9), -1.28e-6, 1.63e8, 0.0, 0.5, htime)

    t_eval = np.arange(0, total_time, dt)
    sol = solve_ivp( DW_EoM, [0, total_time], y0, args=[dw1], t_eval=t_eval)

    h_vals = np.zeros_like(sol.t)
    for i in range(len(h_vals)):
        h_vals[i] = dw1.Happ(t_eval[i])

    return sol.t, sol.y, h_vals, fields, periods
