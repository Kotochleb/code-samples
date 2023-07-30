import numpy as np

class MotorMechanical:
    def __init__(self):
        # Parameters from datasheet in their original units
        self._Jm = 0.39  # Moment of inertia [Kg cm^2]
        self._Tf = 1.2   # Average friction torque [Ncm]
        self._W  = 0.6   # Weight [kg]
        self._D  = 120.0 # Diameter [mm]
        self._LG = 26.0  # Length [mm]    

    @property
    def Jm(self):
        # Moment of inertia [Kg m^2]
        return self._Jm / (100.0**2)

    @property
    def Tf(self):
        # Average friction torque [Nm]
        return self._Tf / 100.0
    
    @property
    def W(self):
        # Weight [kg]
        return self._W
    
    @property
    def D(self):
        # Diameter [m]
        return self._D / 1000.0
    
    @property
    def LG(self):
        # Length [m]
        return self._LG / 1000.0

class MotorPerformance:
    def __init__(self):
        # Parameters from datasheet in their original units
        self._Tp    = 300.0  # Peak torque [Ncm]
        self._Ts    = 20.27  # Stall torque [Ncm]
        self._Tc    = 0.0    # Cogging Torque [Ncm]
        self._T     = 33.0   # Rated continuous torque [Ncm]
        self._Tp_t  = 50.0   # Peak torque time [ms]
        self._P     = 94.0   # Rated power [W]
        self._N     = 3000.0 # Rated speed [rpm]
        self._N_max = 6000.0 # Maximum rated speed [rpm]

    @property
    def Tp(self):
        # Peak torque [Nm]
        return self._Tp / 100.0
    
    @property
    def Ts(self):
        # Stall torque [Nm]
        return self._Ts / 100.0
    
    @property
    def Tc(self):
        # Cogging Torque [Nm]
        return self._Tc / 100.0
    
    @property
    def T(self):
        # Rated continuous torque [Nm]
        return self._T / 100.0
    
    @property
    def Tp_t(self):
        # Peak torque time [s]
        return self._Tp_t * 10**(-9)
    
    @property
    def P(self):
        # Rated power [W]
        return self._P
    
    @property
    def N(self):
        # Rated speed [rad/s]
        return self._N  * (2.0 * np.pi) / (60.0)
    
    @property
    def N_max(self):
        # Maximum rated speed [rad/s]
        return self._N_max  * (2.0 * np.pi) / (60.0)

class MotorThermal:
    def __init__(self):
        # Parameters from datasheet in their original units
        self._RAAR = 2.00 # Thermal resistance at rated speed [°C/W]
        self._RAAS = 2.52 # Thermal resistance at stall [°C/W]

    @property
    def RAAR(self):
        # Thermal resistance at rated speed [°C/W]
        return self._RAAR

    @property
    def RAAS(self):
        # Thermal resistance at stall [°C/W]
        return self._RAAS


class MotorWinding:
    def __init__(self):
        # Parameters from datasheet in their original units
        self._Rm     = 1.1   # Terminal resistance [Ohm]
        self._Rm_err = 0.1   # Plus minus error in winding resistance [# scaled to (0,1)]
        self._Ra     = 0.719 # Armature resistance [Ohm]
        self._Ra_err = 0.1   # Plus minus error in armature resistance [# scaled to (0,1)]
        self._Ke     = 5.0   # Back EMF constant [V/kRPM]
        self._Ke_err = 0.05  # Plus minus error in back EMF constant [# scaled to (0,1)]
        self._Kt     = 4.77  # Torque constant [Ncm/Amp]
        self._Kt_err = 0.05  # Plus minus error in torque constant [# scaled to (0,1)]
        self._Kd     = 0.5   # Viscous Damnp.ping constant [Ncm/kRPM]
        self._L      = 0.03  # Armature inductance [uH]
        self._C      = -0.19 # Temperature coefficient of KE [#/°C]
        self._Z      = 117   # Number of commutation bars [Count]
    

    @property
    def Rm(self):
        # Terminal resistance [Ohm]
        return self._Rm

    @property
    def Rm_low(self):
        # Lower bound of terminal resistance [Ohm]
        return self._Rm * (1 - self._Rm_err)
    
    @property
    def Rm_up(self):
        # Upper bound of terminal resistance [Ohm]
        return self._Rm * (1 + self._Rm_err)
    
    @property
    def Ra(self):
        # Armature resistance [Ohm]
        return self._Ra
    
    @property
    def Ra_low(self):
        # Lower bound of armature resistance [Ohm]
        return self._Ra * (1 - self._Ra_err)
    
    @property
    def Ra_up(self):
        # Upper bound of armature resistance [Ohm]
        return self._Ra * (1 + self._Ra_err)
    
    @property
    def Ke(self):
        # Back EMF constant [V s/rad]
        return self._Ke *  (1 / ((2 * np.pi * 1000) / 60))
    
    @property
    def Ke_low(self):
        # Lower bound of Back EMF constant [V s/rad]
        return self._Ke * (1 - self._Ke_err)
    
    @property
    def Ke_up(self):
        # Upper bound of Back EMF constant [V s/rad]
        return self._Ke * (1 + self._Ke_err)
    
    @property
    def Kt(self):
        # Torque constant [Nm/Amp]
        return self._Kt / 100.0
    
    @property
    def Kt_low(self):
        # Lower bound of torque constant [Ohm]
        return self._Kt * (1 - self._Kt_err)
    
    @property
    def Kt_up(self):
        # Upper bound of torque constant [Ohm]
        return self._Kt * (1 + self._Kt_err)
    
    @property
    def Kd(self):
        # Viscous Damnping constant [Nm s/rad]
        return self._Kd / 100.0 * (1 / ((2.0 * np.pi * 1000.0) / 60.0))
    
    @property
    def L(self):
        # Armature inductance [uH]
        return self._L * 10.0**(-6)
    
    @property
    def C(self):
        # Temperature coefficient of KE [%/°C]
        return self._C
    
    @property
    def Z(self):
        # Number of commutation bars [Count]
        return self._Z