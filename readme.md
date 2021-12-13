# Readme
This code tries to control robot solo to make it walk.
The code is structured as follows:
- First a trajectory optimization problem is solved to get desired CoM and CoP positions
- Then theese trajectories are mapped into the real time controller step time, also trajectories for swing feet are generated using a polynomial interpolation
- Finally a TSID controller tries to follows the trajectories previously generated

The structure of this formulation is the same of the one of  Prof. A. Del Prete in TSID examples.
Also The LMPC_walking library was alredy developed by and here it's just adapted for quadrupeds locomotion.

## To try the code:
- run quad_lipm
- run quad_limpm_to_tsid
- run quad_walking

Unfortunately this doesn't work yet, there are some problems that I still have to figure out

Anyway a much easier problem is shown in com_sine_wave_.py, where the robot is loaded into a PyBullet environment and it tries to follow a sine wave with its CoM
