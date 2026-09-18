import numpy as np
import sys
import os
import gzip
import numpy

# Here are some parameters for the purifying selection model.
#
# issue, we don't really know mu within a factor of 3-10. 
#
# from Zhiru's dN/dS paper, s/mu = 5e03 
#
# From Zhiru's recombination paper, we have
# 
# r/mu = 0.1 (maybe slightly higher)
# dl = 10kb (maybe slightly lower)
# genome size L = 4e06
# lfrac = 1e04/4e06 = 0.003
#
# 2*N*mu >= 2*Tc*mu = 0.01
#
# 
# N*s = 2*N*mu * (s/mu) / 2 > = 2*Tc*mu * (s/mu) / 2 = 25
#
# N*Ud = N*L*mu*fd = L*fd / 2 * (2*N*mu) >= L*fd / 2 * (2*Tc*mu) >= 1e06 * 1e-02 = 1e04
#
# N*R = N*L*r/mu*mu = L / 2 * (r/mu) * (2*N*mu) >= L / 2 * (r/mu) * (2*Tc*mu) = 2e06 * 1e-01 * 1e-02 = 2e03
#
# N >= 25/s = 25/1e-02 = 2500 
# 
# ####
#
# Choose s = 1e-02
# then Ud = fd*L * mu = fd*L * s / (s/mu) = 2e06 * 1e-02 / ( 5e03) = 2e04/(5e03) = 4
# then R = L*r = L*(r/mu)*mu*fd/fd = Ud*(r/mu)/fd = 8*0.1 = 0.8  
# then N can be anything, but >= 25/1e-02 = 2500
# we can go up by a factor of 100 probably. 
#
######
#
# want 1e-02 = 2*mu*Tc = 2*mu*N*Tc/N 
#
# Ud/s = mu*L*fd/s = fd*L / (s/mu) = 2e06 / (s/mu) = 2/5 e03 = 4e02
# 
# Nsd = N*mu * s/mu = (2*Tc*mu) * (s/mu) / (2*Tc/N) = 2.5e01 / (Tc/N)
#
# LEFT OFF!
#
# To get Ud, assume that about ~50 percent of sites in genome are 1-fold degenerate
# so Ud = Utot/2 -> R = 0.2*Ud
# 
# s = 5e03*mu = 
#
# then you can scale Ne*mu 
# 
# from absolute mutation rate, Utot = 4e-04
# genome size L = 4e06
# mu = 1e-10
# 


# ####
#
# Choose s = 1e-02
# then Ud = fd*L * mu = fd*L * s / (s/mu) = 2e06 * 1e-02 / ( 5e03) = 2e04/(5e03) = 4
# then R = L*r = L*(r/mu)*mu*fd/fd = Ud*(r/mu)/fd = 8*0.1 = 0.8  
# then N can be anything, but >= 25/1e-02 = 2500
# we can go up by a factor of 100 probably. 
#
#####

#####
#
# Parameter combinations constrained by empirical data
#
####
sbymu = 5e03 # from Liu & Good 2025 dN/dS decay curve
rbymu = 1e-01 # from Liu & Good 2024 external transfers, using neutral extrapolation
lbyL=0.003 # from Liu & Good 2024 external transfers
L = 4e06 # E. coli genome size
fd = 0.5*0.9 # typical fraction of deleterious sites (1/2 of sites 1D, 90% beneficial)

sd = 1e-02 # arbitrary, sets timescale of simulations. want as large as possible while still in diffusion limit (std(fitness) << 1)
NUn = 500.0 # arbitrary, used for measuring Tc
n = 100 # number of individuals to sample at end 
num_replicates = 10 # number of replicate populations to evolve per simulation 
full_Ns = numpy.logspace(3,5,11)*2.5 # census population sizes to use
Ns = full_Ns
#Ns = full_Ns[0:1] # debug


full_rbymus = [0,rbymu, rbymu*10, rbymu*30] # recombination rates to use
rbymus = full_rbymus
rbymus = [full_rbymus[2]] # debug

# Function that calculates the scaled parameters (NS,NU,NR) from known empirical constraints and simulated values of sd and N
def calculate_scaled_parameters(sd,N,sbymu=sbymu, rbymu=rbymu, lbyL=lbyL, fd=fd, L=L):
	
	Ud = L*fd*sd/sbymu # total genome-wide mutation rate
	R = Ud*rbymu/fd # total genome-wide recombination rate
	
	# calculate scaled parameters
	NS = N*sd
	NU = N*Ud
	NR = N*R
	
	return (NS, NU, NR)

def parse_simulation_output(sd,NS,NU,NR,lbyL):

	simulation_output = []
	current_replicate = []
	
	file = gzip.open("simulation_output_%g_%g_%g_%g_%g.txt.gz" % (sd,NS,NU,NR,lbyL), mode='rt')
	
	file.readline() # header
	for line in file:
		items = line.split(",")
		individual_items = items[0].split(";")
		replicate_idx = int(individual_items[0])
		time = int(individual_items[1])
		fixed_fitness = float(individual_items[2])
		individual_idx = int(individual_items[3])
		individual = (replicate_idx, time, fixed_fitness, individual_idx)
			
		if individual_idx==1 and len(current_replicate)>0:
			# we've started a new sample
			simulation_output.append(current_replicate)
			current_replicate = []
		
		absolute_fitness = float(items[1])
		synonymous_mutations = [float(subitem) for subitem in items[2].split()]
		nonsynonymous_mutations = [float(subitem) for subitem in items[3].split()]
		current_replicate.append((individual, absolute_fitness, synonymous_mutations, nonsynonymous_mutations))
	
	simulation_output.append(current_replicate)
	return simulation_output
	
	
if __name__=='__main__':
	
	for N in Ns:
		for rbymu in rbymus:
	
			Un = NUn/N
			t_max = 10*N
		
			NS,NU,NR = calculate_scaled_parameters(sd,N,rbymu=rbymu)
			
			Ud = NU/N
			R = NR/N
			
			#usage: ./simulation n_replicates t_max n N Usyn Unon s R lfrac
			sys.stderr.write("Simulating N=%g, sd=%g, rbymu=%g...\n" % (N,sd,rbymu))
			sys.stderr.write('./simulation %d %d %d %g %g %g %g %g %g | gzip -c > simulation_output_%g_%g_%g_%g_%g.txt.gz\n' % (num_replicates, t_max, n, N, Un, Ud, -sd,R,lbyL,sd,NS,NU,NR,lbyL))
			os.system('./simulation %d %d %d %g %g %g %g %g %g | gzip -c > simulation_output_%g_%g_%g_%g_%g.txt.gz' % (num_replicates, t_max, n, N, Un, Ud, -sd,R,lbyL,sd,NS,NU,NR,lbyL))
		
		
			sys.stderr.write("Done!\n")