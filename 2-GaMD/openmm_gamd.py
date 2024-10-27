#!/usr/bin/env python
from __future__ import division, print_function

import os, sys

# Import the OpenMM modules that are necessary
from openmm.app import *
from openmm import *
from openmm.unit import *

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

"""
GaMD integrator for the total energy
"""

class GaMDIntegrator(CustomIntegrator):
    def __init__(self, temperature, friction, dt, ktot, Etot):
        self.ktot = ktot 
        self.Etot = Etot 
        
        CustomIntegrator.__init__(self, dt)
            #new added:
        self.addGlobalVariable("ktot", self.ktot)
        self.addGlobalVariable("Etot", self.Etot)
        
            #normal langevin:  
        self.addGlobalVariable("temperature", temperature);
        self.addGlobalVariable("friction", friction);
        self.addGlobalVariable("vscale", 0);
        self.addGlobalVariable("fscale", 0);
        self.addGlobalVariable("noisescale", 0);
        self.addPerDofVariable("x0", 0);
              
            #normal langevin:                                                                  
        self.addUpdateContextState();
                
        self.addComputeGlobal("vscale", "exp(-dt*friction)");
        self.addComputeGlobal("fscale", "(1-vscale)/friction");
        #original line:                
        self.addComputeGlobal("noisescale", "sqrt(kT*(1-vscale*vscale)); kT=0.00831451*temperature");
        self.addComputePerDof("x0", "x");
            #original langevin line:                                                                                      
        #self.addComputePerDof("v", "vscale*v + fscale*f/m + noisescale*gaussian/sqrt(m)");  
            #GaMD:
        dof_string = "vscale*v + fscale*fprime/m + noisescale*gaussian/sqrt(m);"
        dof_string+= "fprime = f*((1-modifyTotal) + modifyTotal* (1 - ktot*(Etot - energy)) );"
        dof_string+= "modifyTotal=step(Etot-energy);"
        self.addComputePerDof("v", dof_string);

            #normal langevin                                            
        self.addComputePerDof("x", "x+dt*v");
        self.addConstrainPositions();
        self.addComputePerDof("v", "(x-x0)/dt");
        #self.addComputePerDof("veloc", "v")
 
    def getKtot(self):
        return self.getGlobalVariableByName('ktot')/kilojoules_per_mole
       
    def setKtot(self, newK):
        if not is_quantity(newK):
            newK = newK/kilojoules_per_mole
        self.setGlobalVariableByName('ktot', newK)

    def getEtot(self):
        return self.getGlobalVariableByName('Etot')*kilojoules_per_mole
       
    def setEtot(self, newE):
        if not is_quantity(newE):
            newE = newE*kilojoules_per_mole
        self.setGlobalVariableByName('Etot', newE)
                  
    def getTotBoost(self, totEnergy):
        ktot = self.getGlobalVariableByName('ktot')/kilojoules_per_mole
        Etot = self.getGlobalVariableByName('Etot')*kilojoules_per_mole
        if not is_quantity(totEnergy):
            totEnergy = totEnergy*kilojoules_per_mole # Assume kJ/mole
        if (totEnergy > Etot):
            return 0*kilojoules_per_mole #no boosting
        return ( 0.5 * ktot * (Etot-totEnergy)**2 ) # 'k' parameter should instead be per kj/mol
        
    def getEffectiveEnergy(self, totEnergy):
        if not is_quantity(totEnergy):
            totEnergy = totEnergy*kilojoules_per_mole # Assume kJ/mole
        
        total_boost = self.getTotBoost(totEnergy)
        return totEnergy + total_boost
 
"""
GaMD integrator for group energy
"""

class GaMDForceIntegrator(CustomIntegrator):
    def __init__(self, temperature, friction, dt, kgrp, Egrp, forceGroup):
        self.kgrp = kgrp
        self.Egrp = Egrp
        self.forceGroup = str(forceGroup)
        
        CustomIntegrator.__init__(self, dt)
            #new added:
        self.addGlobalVariable("kgrp", self.kgrp)
        self.addGlobalVariable("Egrp", self.Egrp)
        self.addGlobalVariable("groupEnergy", 0)
        
            #normal langevin:  
        self.addGlobalVariable("temperature", temperature);
        self.addGlobalVariable("friction", friction);
        self.addGlobalVariable("vscale", 0);
        self.addGlobalVariable("fscale", 0);
        self.addGlobalVariable("noisescale", 0);
        self.addPerDofVariable("x0", 0);
        
        self.addPerDofVariable("fgrp", 0)
        
            #normal langevin:                                                                  
        self.addUpdateContextState();
        
        self.addComputeGlobal("groupEnergy", "energy"+self.forceGroup)
        self.addComputePerDof("fgrp", "f"+self.forceGroup)
        
        self.addComputeGlobal("vscale", "exp(-dt*friction)");
        self.addComputeGlobal("fscale", "(1-vscale)/friction");
        #original line:                
        self.addComputeGlobal("noisescale", "sqrt(kT*(1-vscale*vscale)); kT=0.00831451*temperature");
        self.addComputePerDof("x0", "x");
            #original langevin line:                                                                                      
        #self.addComputePerDof("v", "vscale*v + fscale*f/m + noisescale*gaussian/sqrt(m)");  
            #GaMD:
        dof_string = "vscale*v + fscale*fprime/m + noisescale*gaussian/sqrt(m);"
        dof_string+= "fprime= fother + fprime1;"
        dof_string+= "fprime1 = fgrp*((1-modifyGroup) + modifyGroup* (1 - kgrp*(Egrp - groupEnergy)) ) ;"       
        dof_string+= "fother=f-fgrp;"
        dof_string+= "modifyGroup=step(Egrp-groupEnergy);"
        self.addComputePerDof("v", dof_string); 

            #normal langevin                                            
        self.addComputePerDof("x", "x+dt*v");
        self.addConstrainPositions();
        self.addComputePerDof("v", "(x-x0)/dt");
 
    def getKgrp(self):
        return self.getGlobalVariableByName('kgrp')/kilojoules_per_mole
     
    def setKgrp(self, newK):
        if not is_quantity(newK):
            newK = newK/kilojoules_per_mole
        self.setGlobalVariableByName('kgrp', newK)

    def getEgrp(self):
        return self.getGlobalVariableByName('Egrp')*kilojoules_per_mole
      
    def setEgrp(self, newE):
        if not is_quantity(newE):
            newE = newE*kilojoules_per_mole
        self.setGlobalVariableByName('Egrp', newE)
 
    def getGrpBoost(self, grpEnergy):
        kgrp = self.getGlobalVariableByName('kgrp')/kilojoules_per_mole
        Egrp = self.getGlobalVariableByName('Egrp')*kilojoules_per_mole
        if not is_quantity(grpEnergy):
            grpEnergy = grpEnergy*kilojoules_per_mole # Assume kJ/mole
        if (grpEnergy > Egrp):
            return 0*kilojoules_per_mole #no boosting
        return ( 0.5 * kgrp * (Egrp-grpEnergy)**2 ) # 'k' parameter should instead be per kj/mol
    
    def getEffectiveEnergy(self, grpEnergy):
        if not is_quantity(grpEnergy):
            grpEnergy = grpEnergy*kilojoules_per_mole # Assume kJ/mole
        
        group_boost = self.getGrpBoost(grpEnergy)
        return totEnergy + group_boost

"""
Dual GaMD integrator, total boost plus group boost
"""

class DualGaMDIntegrator(CustomIntegrator):
    def __init__(self, temperature, friction, dt, ktot, Etot, kgrp, Egrp, forceGroup):
        self.ktot = ktot 
        self.Etot = Etot 
        self.kgrp = kgrp
        self.Egrp = Egrp
        self.forceGroup = str(forceGroup)
        
        CustomIntegrator.__init__(self, dt)
            #lew added:
        self.addGlobalVariable("ktot", self.ktot)
        self.addGlobalVariable("Etot", self.Etot)
        self.addGlobalVariable("kgrp", self.kgrp)
        self.addGlobalVariable("Egrp", self.Egrp)
        self.addGlobalVariable("groupEnergy", 0)
        
            #normal langevin:  
        self.addGlobalVariable("temperature", temperature);
        self.addGlobalVariable("friction", friction);
        self.addGlobalVariable("vscale", 0);
        self.addGlobalVariable("fscale", 0);
        self.addGlobalVariable("noisescale", 0);
        self.addPerDofVariable("x0", 0);
        
        self.addPerDofVariable("fgrp", 0)
        
            #normal langevin:                                                                  
        self.addUpdateContextState();
        
        self.addComputeGlobal("groupEnergy", "energy"+self.forceGroup)
        self.addComputePerDof("fgrp", "f"+self.forceGroup)
        
        self.addComputeGlobal("vscale", "exp(-dt*friction)");
        self.addComputeGlobal("fscale", "(1-vscale)/friction");
        #original line:                
        self.addComputeGlobal("noisescale", "sqrt(kT*(1-vscale*vscale)); kT=0.00831451*temperature");
        self.addComputePerDof("x0", "x");
            #original langevin line:                                                                                      
        #self.addComputePerDof("v", "vscale*v + fscale*f/m + noisescale*gaussian/sqrt(m)");  
            #GaMD:
        dof_string = "vscale*v + fscale*fprime/m + noisescale*gaussian/sqrt(m);"
        dof_string+= "fprime= fprime1 + fprime2;"
        #fprime2 is the dihedral force modified by the boost. Boost calculated using group only. 
        dof_string+= "fprime2 = fgrp*((1-modifyGroup) + modifyGroup* (1 - kgrp*(Egrp - groupEnergy)) ) ;"
        #fprime1 is the other forces modified by the boost. Boost calculated for the energy without the group energy.
        dof_string+= "fprime1 = fother*((1-modifyOther) + modifyOther* (1 - ktot*(Etot - energy + groupEnergy)) );"
        dof_string+= "fother=f-fgrp;"
        dof_string+= "modifyGroup=step(Egrp-groupEnergy);"
        dof_string+= "modifyOther=step(Etot-energy+groupEnergy);"
        self.addComputePerDof("v", dof_string); 
            #normal langevin                                            
        self.addComputePerDof("x", "x+dt*v");
        self.addConstrainPositions();
        self.addComputePerDof("v", "(x-x0)/dt");
        #self.addComputePerDof("veloc", "v")
 
    def getKtot(self):
        return self.getGlobalVariableByName('ktot')/kilojoules_per_mole

    def setKtot(self, newK):
        if not is_quantity(newK):
            newK = newK/kilojoules_per_mole
        self.setGlobalVariableByName('ktot', newK)
 
    def getEtot(self):
        return self.getGlobalVariableByName('Etot')*kilojoules_per_mole

    def setEtot(self, newE):
        if not is_quantity(newE):
            newE = newE*kilojoules_per_mole
        self.setGlobalVariableByName('Etot', newE)

    def getKgrp(self):
        return self.getGlobalVariableByName('kgrp')/kilojoules_per_mole

    def setKgrp(self, newK):
        if not is_quantity(newK):
            newK = newK/kilojoules_per_mole
        self.setGlobalVariableByName('kgrp', newK)

    def getEgrp(self):
        return self.getGlobalVariableByName('Egrp')*kilojoules_per_mole
        
    def setEgrp(self, newE):
        if not is_quantity(newE):
            newE = newE*kilojoules_per_mole
        self.setGlobalVariableByName('Egrp', newE)
          
    def getGrpBoost(self, grpEnergy):
        kgrp = self.getGlobalVariableByName('kgrp')/kilojoules_per_mole
        Egrp = self.getGlobalVariableByName('Egrp')*kilojoules_per_mole
        if not is_quantity(grpEnergy):
            grpEnergy = grpEnergy*kilojoules_per_mole # Assume kJ/mole
        if (grpEnergy > Egrp):
            return 0*kilojoules_per_mole #no boosting
        return ( 0.5 * kgrp * (Egrp-grpEnergy)**2 ) # 'k' parameter should instead be per kj/mol
    
    def getTotBoost(self, totEnergy):
        ktot = self.getGlobalVariableByName('ktot')/kilojoules_per_mole
        Etot = self.getGlobalVariableByName('Etot')*kilojoules_per_mole
        if not is_quantity(totEnergy):
            totEnergy = totEnergy*kilojoules_per_mole # Assume kJ/mole
        if (totEnergy > Etot):
            return 0*kilojoules_per_mole #no boosting
        return ( 0.5 * ktot * (Etot-totEnergy)**2 ) # 'k' parameter should instead be per kj/mol
        
    def getEffectiveEnergy(self, totEnergy, grpEnergy):
        if not is_quantity(totEnergy):
            totEnergy = totEnergy*kilojoules_per_mole # Assume kJ/mole
        if not is_quantity(grpEnergy):
            grpEnergy = grpEnergy*kilojoules_per_mole # Assume kJ/mole
        
        group_boost = self.getGrpBoost(grpEnergy)
        total_boost = self.getTotBoost(totEnergy)
        
        return totEnergy + group_boost + total_boost
        
#These functions help calculate the k and E parameters based on Miao's rules

def get_statistics(pe_trace, boost_trace=0):
    arr = np.array(pe_trace)
    arr = arr + np.array(boost_trace)
    Vmax = arr.max()
    Vmin = arr.min()
    Vavg = arr.mean()
    Vstd = np.std(arr)
    
    return [Vmax, Vmin, Vavg, Vstd]

TEMPERATURE = 300*kelvin
SIGMA0 = (MOLAR_GAS_CONSTANT_R * TEMPERATURE ).value_in_unit(kilojoule_per_mole)

def calc_parameters(Vmax, Vmin, Vavg, Vstd, mode='low', sigma_0=SIGMA0, verbose=False):
    E = Vmax
    k_0 = min(1, (sigma_0/Vstd) * ((Vmax-Vmin)/(Vmax-Vavg)))
    k = k_0 * (1 / (Vmax - Vmin) )
    
    if verbose:
        return E, k, k_0
    else:
        return E, k

def report(sim, prod):
    grp_nrg = sim.context.getState(getEnergy=True, groups={5}).getPotentialEnergy()/kilojoules_per_mole
    tot_nrg = sim.context.getState(getEnergy=True,).getPotentialEnergy()/kilojoules_per_mole
        
    grp_boost = sim.integrator.getGrpBoost(grp_nrg)/kilojoule_per_mole
    tot_boost = sim.integrator.getTotBoost(tot_nrg)/kilojoule_per_mole
        
    prod['grp_boosts'].append(grp_boost)
    prod['tot_boosts'].append(tot_boost)
        
    prod['grp_nrgs'].append(grp_nrg)
    prod['tot_nrgs'].append(tot_nrg)

#using Miao's anharmonicity calculator:
def anharm(data):
    var=np.var(data)
    # hist, edges=np.histogram(data, 50, normed=True)
    hist, edges=np.histogram(data, 50, density=True)
    hist=np.add(hist,0.000000000000000001)  ###so that distrib
    dx=edges[1]-edges[0]
    S1=-1*np.trapz(np.multiply(hist, np.log(hist)),dx=dx)
    S2=0.5*np.log(np.add(2.00*np.pi*np.exp(1)*var,0.000000000000000001))
    alpha=S2-S1
    if np.isinf(alpha):
       alpha = 100
    return alpha


