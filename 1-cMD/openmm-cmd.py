#!/usr/bin/env python
from __future__ import division, print_function

# Import the OpenMM modules that are necessary
from openmm.app import *
from openmm import *

# Import the ParmEd/OpenMM modules
from parmed.amber import AmberParm, Rst7, AmberMdcrd, AmberMask, NetCDFTraj
from parmed.openmm import (StateDataReporter, NetCDFReporter, MdcrdReporter,
        RestartReporter, ProgressReporter, EnergyMinimizerReporter)
from parmed import unit as u
from parmed.utils.six.moves import range

# Import other modules
from optparse import OptionParser
import random

parser = OptionParser("Usage: openmm_cmd.py -o output_file -p prmtop_file -c inpcrd_file -r restart_file -x traj_file")
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
(options, args) = parser.parse_args()

TEMPERATURE = 300*u.kelvin

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

# Retrieve the NonbondedForce
for force in system.getForces():
    if isinstance(force, NonbondedForce):
        nbforce = force

# Keep consistent with the LiGaMD simulations
nbforce.setUseDispersionCorrection(False)

# Set the integrator and simulation
integrator = LangevinIntegrator(TEMPERATURE, 2.0/u.picosecond, 0.001*u.picosecond)
integrator.setConstraintTolerance(1e-05)
platform = Platform.getPlatformByName("CUDA")
sim = Simulation(parm.topology, system, integrator, platform)

# Set the positions and box vectors
sim.context.setPositions(parm.positions)
sim.context.setPeriodicBoxVectors(*parm.box_vectors)

# Print the initial energy
print("Initial energy:")
state = sim.context.getState(getEnergy=True)
print('Total Energy: ', state.getPotentialEnergy())

# Set random velocities
rn = random.randint(0, 10000)
sim.context.setVelocitiesToTemperature(TEMPERATURE, rn)

# Set number of steps for MD
md_steps = 50000000 # 50 ns

# Add the state data reporters
# 1. mdout reporter
mdout_rep = StateDataReporter(options.out, 1000, volume=True, density=False)
sim.reporters.append(mdout_rep)

# 2. mdinfo reporter
mdinfo_rep = ProgressReporter("mdinfo", 1000, md_steps, volume=True, density=False)
sim.reporters.append(mdinfo_rep)

# 3. netcdf reporter
netcdf_rep = NetCDFReporter(options.traj, 1000, crds=True, vels=True, frcs=False)
sim.reporters.append(netcdf_rep)

# 4. restart reporter
rst_rep = RestartReporter(options.restart, 1000, False, False)
sim.reporters.append(rst_rep)

# Production simulations
sim.step(md_steps)

#
# Report the final state
#
print("At Final Stage...")

# Save the structure into the rst file
final_state = sim.context.getState(getPositions=True,
     getVelocities=True, enforcePeriodicBox=True)
rst_rep.report(sim, final_state)

# Print final energy
state = sim.context.getState(getEnergy=True)
print('Total Energy: ', state.getPotentialEnergy())

quit()

