#!/usr/bin/env python
from __future__ import division, print_function

# Import the OpenMM modules that are necessary
from openmm.app import *
from openmm import *

# Import the GaMD module for OpenMM
import sys
from openmm_gamd import *

# Import the Amber/OpenMM modules
from parmed.amber import AmberParm, Rst7, AmberMdcrd, AmberMask, NetCDFTraj
from parmed.openmm import (StateDataReporter, NetCDFReporter, MdcrdReporter,
        RestartReporter, ProgressReporter, EnergyMinimizerReporter)
from parmed import unit as u
from parmed.utils.six.moves import range

# Import other modules
from optparse import OptionParser
import random

parser = OptionParser("Usage: gamd_eq-rf.py -o output_file -p prmtop_file -c inpcrd_file -r restart_file -x traj_file -l lig_range")
parser.add_option("-o", dest="out", type='string',
                  help="Output file name")
parser.add_option("-p", dest="prmtop", type='string',
                  help="Prmtop file name")
parser.add_option("-c", dest="crd", type='string',
                  help="Inpcrd file name")
parser.add_option("-r", dest="restart", type='string',
                  help="Restart file name")
parser.add_option("-x", dest="traj", type='string',
                  help="Trajectory file name")
parser.add_option("-l", dest="lig", type='string',
                  help="Ligand index range")
(options, args) = parser.parse_args()

TEMPERATURE = 300*u.kelvin
SIGMA0 = (u.MOLAR_GAS_CONSTANT_R * TEMPERATURE ).value_in_unit(u.kilojoule_per_mole) 

# Load the Amber topology file
parm = AmberParm(options.prmtop, options.crd)
system = parm.createSystem(nonbondedMethod=CutoffPeriodic,
                nonbondedCutoff=10.0*u.angstrom,
                constraints=HBonds, rigidWater=True,
                implicitSolvent=None,
                implicitSolventKappa=0.0*(1.0/u.angstrom),
                soluteDielectric=1.0,
                solventDielectric=78.5,
                removeCMMotion=True,
                ewaldErrorTolerance=5e-05,
                flexibleConstraints=False,
                verbose=False,
)

# 1. Make two sets of particles, one that contains the ligand
# and the other which contains all the other particles.
lig_index = options.lig.split('-')
lig_index = [int(i) for i in lig_index]
i_start = lig_index[0] - 1 # note that OpenMM starts from atom 0, while pdb starts from 1
i_end = lig_index[-1] 

# Print out the ligand atoms you have chosen
ligand_particles = range(i_start, i_end)
for i in ligand_particles:
    print("The ligand you have chosen contains atom: ", parm.atoms[i].name)

ligand_particles = set(ligand_particles)
all_particles = set(range(system.getNumParticles()))
other_particles = all_particles - ligand_particles

# 2. Add the interaction between the ligand and environment as a Customized Force
# Retrieve the NonbondedForce
for force in system.getForces():
    if isinstance(force, NonbondedForce):
        nbforce = force

# To be consistent with the CustomNonbondedForce
nbforce.setUseDispersionCorrection(False)
eps_solvent = nbforce.getReactionFieldDielectric()

# Create a CustomNonbondedForce using the same parameters from the nbforce

# Add the energy expression
#ONE_4PI_EPS0 = 138.935456 kJ/mol * nm / e^2
# r is in unit of nm
ONE_4PI_EPS0 = 138.935456
cutoff = 1.0 * u.nanometer # unit nm
krf = (1/ (cutoff**3)) * (eps_solvent - 1) / (2*eps_solvent + 1)
crf = (1/ cutoff) * (3* eps_solvent) / (2*eps_solvent + 1)

energy_expression  = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod*(1/r + krf*r*r - crf);"
energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
energy_expression += "sigma = 0.5*(sigma1+sigma2);"
energy_expression += "chargeprod = charge1*charge2;"
energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
energy_expression += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
energy_expression += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
lig_env_force = CustomNonbondedForce(energy_expression)

# Setting per particle parameters
lig_env_force.addPerParticleParameter('charge')
lig_env_force.addPerParticleParameter('sigma')
lig_env_force.addPerParticleParameter('epsilon')

# Cutoff settings
lig_env_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
lig_env_force.setCutoffDistance(10.0*u.angstrom)

# Add per particle parameters
for index in range(system.getNumParticles()):
    [charge, sigma, epsilon] = nbforce.getParticleParameters(index)
    lig_env_force.addParticle([charge, sigma, epsilon])

# Making the customized nonbonded force only having the interaction group
lig_env_force.addInteractionGroup(ligand_particles, all_particles)

# Setting the exclusion group for the customized force group, which should be the same as the nbforce group
# The exception is about 1-2, 1-3 and 1-4 interactions
for ii in range(nbforce.getNumExceptions()):
    i, j, qq, ss, ee = nbforce.getExceptionParameters(ii)
    lig_env_force.addExclusion(i, j)

# Add the force between ligand and environment (protein and solvent)
system.addForce(lig_env_force)

# 3. Remove the ligand and environment interactions from the current nonbonded force
# set the values of charge, sigma and epsilon by copying them from the existing NonBondedForce
for index in ligand_particles:
   # remove the ligand particles from the existing NonBondedForce
   [charge, sigma, epsilon] = nbforce.getParticleParameters(index)
   nbforce.setParticleParameters(index, charge*0, sigma*0, epsilon*0)

# Set the force group of CustomNB as 5, and all others as 0
for f in system.getForces():
    if isinstance(f, CustomNonbondedForce):
        print('Found the CustomNonbondedForce - setting group to 5')
        f.setForceGroup(5)
    else:
        f.setForceGroup(0)
    print(f.getForceGroup(), f.__class__)

integrator = DualGaMDIntegrator(TEMPERATURE, 2.0/u.picosecond, 0.001*u.picosecond, 
                    1/u.kilojoules_per_mole,
                    -1e6*u.kilojoules_per_mole, #set a ridiculous low threshold to start, which means no boost
                    1/u.kilojoules_per_mole,
                    -1e6*u.kilojoules_per_mole, #set a ridiculous low threshold to start, which means no boost
                    5)

integrator.setConstraintTolerance(1e-05)
platform = Platform.getPlatformByName("CUDA")
sim = Simulation(parm.topology, system, integrator, platform)

# Set the positions and box vectors
sim.context.setPositions(parm.positions)
sim.context.setPeriodicBoxVectors(*parm.box_vectors)

# Print different force groups
print("Initial energy:")

state = sim.context.getState(getEnergy=True, groups={5})
print('CustomNonbonded Energy: ', state.getPotentialEnergy())

state = sim.context.getState(getEnergy=True, groups={0})
print('Other Energy: ', state.getPotentialEnergy())

state = sim.context.getState(getEnergy=True)
print('Total Energy: ', state.getPotentialEnergy())

# Set velocities
rn = random.randint(0, 10000)
sim.context.setVelocitiesToTemperature(TEMPERATURE, rn)

# Add the state data reporters
# 1. mdout reporter
mdout_rep = StateDataReporter(options.out, 1000, volume=True, density=False)
sim.reporters.append(mdout_rep)

# 2. mdinfo reporter
mdinfo_rep = ProgressReporter("mdinfo", 1000, 2450000, volume=True, density=False)
sim.reporters.append(mdinfo_rep)

# 3. netcdf reporter
netcdf_rep = NetCDFReporter(options.traj, 1000, crds=True, vels=True, frcs=False)
sim.reporters.append(netcdf_rep)

# 4. restart reporter
rst_rep = RestartReporter(options.restart, 1000, False, False)
sim.reporters.append(rst_rep)

#
# Stage A: Collect statistics from cMD, only get the energies (E and Egrp), not the boost
#
print("At Stage A...")

grp_nrgs = list()
tot_nrgs = list()

# equil for 200 ps
sim.step(200000)

# Run for 200 ps
for i in range(2000):
    sim.step(100)

    grp_nrg = sim.context.getState(getEnergy=True, groups={5}).getPotentialEnergy()/u.kilojoule_per_mole
    tot_nrg = sim.context.getState(getEnergy=True, groups={0}).getPotentialEnergy()/u.kilojoule_per_mole
 
    print(i, "grp_nrg =" , grp_nrg, "tot_nrg =", tot_nrg)

    grp_nrgs.append(grp_nrg)
    tot_nrgs.append(tot_nrg)

#if doing total boost as well:
Vmax, Vmin, Vavg, Vstd = get_statistics(tot_nrgs,0)
E,k = calc_parameters(Vmax, Vmin, Vavg, Vstd, sigma_0=6*SIGMA0)
print('E:',E)
sim.integrator.setEtot(E)
sim.integrator.setKtot(k)

#for the group boost:
Vmax, Vmin, Vavg, Vstd = get_statistics(grp_nrgs,0)
E,k = calc_parameters(Vmax, Vmin, Vavg, Vstd, sigma_0=6*SIGMA0)
print('Egrp:',E)
sim.integrator.setEgrp(E)
sim.integrator.setKgrp(k)

#
# Stage B: Run iterative GaMD to get k and E parameters
#
print("At Stage B...")

grp_nrgs = list()
tot_nrgs = list()

grp_boosts = list()
tot_boosts = list()

ks = list()
k_0s = list()
vmaxs = list()

gks = list()
gk_0s = list()
gvmaxs = list()

# Running for 50 * 41 = 2050 ps

for _ in range(50):

    #equil for 1ps
    sim.step(1000)
    
    #run for 40ps, collecting data every 0.2 ps
    for i in range(200):
        sim.step(200)
        
        grp_nrg = sim.context.getState(getEnergy=True, groups={5}).getPotentialEnergy()/u.kilojoules_per_mole
        tot_nrg = sim.context.getState(getEnergy=True, groups={0}).getPotentialEnergy()/u.kilojoules_per_mole
        
        grp_boost = sim.integrator.getGrpBoost(grp_nrg)/u.kilojoule_per_mole
        tot_boost = sim.integrator.getTotBoost(tot_nrg)/u.kilojoule_per_mole
        
        grp_boosts.append(grp_boost)
        tot_boosts.append(tot_boost)
        
        grp_nrgs.append(grp_nrg)
        tot_nrgs.append(tot_nrg)
              
    #For the total boost
    Vmax, Vmin, Vavg, Vstd = get_statistics(tot_nrgs, 0)
    print("Vmax, Vmin, Vavg, Vstd:", Vmax, Vmin, Vavg, Vstd)
    E,k,k_0 = calc_parameters(Vmax, Vmin, Vavg, Vstd, sigma_0 = 6*SIGMA0,verbose=True)
    sim.integrator.setEtot(E)
    sim.integrator.setKtot(k)
  
    ks.append(k)
    k_0s.append(k_0)
    vmaxs.append(E)
    print("total boost")
    print("E, k, k0:", E, k, k_0)

    #For the group boost
    Vmax, Vmin, Vavg, Vstd = get_statistics(grp_nrgs, 0)
    print("Vmax, Vmin, Vavg, Vstd:", Vmax, Vmin, Vavg, Vstd)
    E,k,k_0 = calc_parameters(Vmax, Vmin, Vavg, Vstd, sigma_0 = 6*SIGMA0, verbose=True)
    sim.integrator.setEgrp(E)
    sim.integrator.setKgrp(k)

    gks.append(k)
    gk_0s.append(k_0)
    gvmaxs.append(E)
    print("group boost")
    print("E, k, k0:", E, k, k_0)

# Get the final parameters
Etot = sim.integrator.getEtot()
ktot = sim.integrator.getKtot()
Egrp = sim.integrator.getEgrp()
kgrp = sim.integrator.getKgrp()

w_file = open('gamd-para.txt', 'w')
print("# Final GaMD parameters:", file=w_file)
print("Etot = ", Etot, file=w_file) # unit is kJ/mol
print("ktot = ", ktot, file=w_file) # unit is mol/kJ
print("Egrp = ", Egrp, file=w_file) # unit is kJ/mol
print("kgrp = ", kgrp, file=w_file) # unit is mol/kJ
w_file.close()

quit()
