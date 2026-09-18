import gzip
import numpy
import pylab

import matplotlib.colors as colors
import matplotlib.cm as cmx
from math import log10,ceil
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from numpy.random import randint,random_sample,shuffle
import matplotlib.colors as mcolors

mpl.rcParams['font.size'] = 8
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

import run_simulation

#sd = 1e-02 #run_simulation.sd
sbymus = [5e03]
rbymus = run_simulation.full_rbymus
Nss = [run_simulation.full_Ns[0:8],run_simulation.full_Ns,run_simulation.full_Ns[0:4],run_simulation.full_Ns[0:5]] 
sd = 1e-02
NUn = run_simulation.NUn
#sds = [1e-02]
lbyL = run_simulation.lbyL
sbymu = run_simulation.sbymu
# r*ell/s = r/mu * (mu/s) * ell = 0.1 * 1e04 / (5e03) = 0.2

R_idxs = range(0,len(rbymus))

simulation_results = {}
for sbymu_idx in range(0,len(sbymus)):
	sbymu = sbymus[sbymu_idx]
	for R_idx in R_idxs:
		rbymu = rbymus[R_idx]
		Ns = Nss[R_idx]
		simulation_results[(sbymu_idx,R_idx)] = {}
		for N in Ns:
	
			NS,NU,NR = run_simulation.calculate_scaled_parameters(sd,N,sbymu=sbymu,rbymu=rbymu)
				
			print(sd,N,NS,NU,NR)
			simulation_results[(sbymu_idx,R_idx)][N] = run_simulation.parse_simulation_output(sd,NS,NU,NR,lbyL)
	
syn_pis = {}
Npfixs = {}

for sbymu_idx,R_idx in simulation_results.keys():
	sbymu = sbymus[sbymu_idx]
	rbymu = rbymus[R_idx]
	Ns = Nss[R_idx]
	
	syn_pis[(sbymu_idx,R_idx)] = {}
	Npfixs[(sbymu_idx,R_idx)] = {}
	
	for N in Ns:
	
		NS,NU,NR = run_simulation.calculate_scaled_parameters(sd,N,sbymu=sbymu,rbymu=rbymu)	
	
		syn_pis[(sbymu_idx,R_idx)][N] = []
		Npfixs[(sbymu_idx,R_idx)][N] = []
	
		for replicate_idx in range(0,len(simulation_results[(sbymu_idx,R_idx)][N])):
		
			synonymous_mutation_counts = {}
			avg_fitness = 0
			time = simulation_results[(sbymu_idx,R_idx)][N][replicate_idx][0][0][1]
			
			# sample size
			n = len(simulation_results[(sbymu_idx,R_idx)][N][replicate_idx])
			
			for individual_idx in range(0,len(simulation_results[(sbymu_idx,R_idx)][N][replicate_idx])):
			
				absolute_fitness = simulation_results[(sbymu_idx,R_idx)][N][replicate_idx][individual_idx][1]
				
				avg_fitness += absolute_fitness
				
				synonymous_mutations = simulation_results[(sbymu_idx,R_idx)][N][replicate_idx][individual_idx][2]
				for mutation in synonymous_mutations:
					if mutation not in synonymous_mutation_counts:
						synonymous_mutation_counts[mutation] = 0
					synonymous_mutation_counts[mutation]+=1
			
			avg_fitness = avg_fitness*1.0/n
			
			# Calculate Npfix = v/Us
			Npfix = -1*avg_fitness*N*N/NU/NS/time
			Npfixs[(sbymu_idx,R_idx)][N].append(Npfix)
				
			# calculate heterozygosity (pi) for sample
			syn_pi = 0
			for mutation in synonymous_mutation_counts:
				k = synonymous_mutation_counts[mutation]
				syn_pi += 2.0*k*(n-k)/(n*(n-1))
			
			syn_pis[(sbymu_idx,R_idx)][N].append(syn_pi)
	
		syn_pis[(sbymu_idx,R_idx)][N] = numpy.array(syn_pis[(sbymu_idx,R_idx)][N])
		Npfixs[(sbymu_idx,R_idx)][N] = numpy.array(Npfixs[(sbymu_idx,R_idx)][N])
		

######################################
#
# Done loading data. Set up figure!
#
######################################
fig = pylab.figure(figsize=(7.42, 2))

outer_grid = gridspec.GridSpec(1, 2, width_ratios = [1,0.1], wspace=0.1)

# Set up plot to hold a legend
inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1,1],wspace=0.4,subplot_spec=outer_grid[0])



legend_axis = pylab.Subplot(fig, outer_grid[1])
fig.add_subplot(legend_axis)
legend_axis.set_ylim([0,1])
legend_axis.set_xlim([0,1])
legend_axis.spines['top'].set_visible(False)
legend_axis.spines['right'].set_visible(False)
legend_axis.spines['left'].set_visible(False)
legend_axis.spines['bottom'].set_visible(False)
legend_axis.set_xticks([])
legend_axis.set_yticks([])  

tc_axis = pylab.Subplot(fig, inner_grid[0])
fig.add_subplot(tc_axis)
tc_axis.set_ylabel('$T_{mrca} \cdot \mu$')
tc_axis.set_xlabel('$N \cdot \\mu$')
tc_axis.set_xlim([2e-03,2])
tc_axis.set_ylim([1e-05,3e-02])
tc_axis.plot([2e-03,2],[1e-02,1e-02],'k:')
tc_axis.plot([2e-03,2],[1/sbymu,1/sbymu],'k:')
tc_axis.plot([2e-03,2],[3e-05,3e-05],'k:')

pfix_axis = pylab.Subplot(fig, inner_grid[1])
fig.add_subplot(pfix_axis)
pfix_axis.set_ylabel('$N p_\mathrm{fix}(-s)$')
pfix_axis.set_xlabel('$T_\mathrm{mrca} \cdot s$')	
pfix_axis.set_xlim([0,7])
pfix_axis.set_ylim([1e-03,2])
theory_tcss = numpy.linspace(0.01,10,50)
good_etal_npfixs = numpy.exp(-theory_tcss/2)
kimura_npfixs = 2*theory_tcss*numpy.exp(-2*theory_tcss)/(1-numpy.exp(-2*theory_tcss))
pfix_axis.semilogy(theory_tcss,good_etal_npfixs,'k:')	
pfix_axis.semilogy(theory_tcss,kimura_npfixs,'k:')


symbol_idx_map = {0:'o',1:'s'}
for sbymu_idx,R_idx in syn_pis.keys():
	sbymu = sbymus[sbymu_idx]
	rbymu = rbymus[R_idx]
	Ns = Nss[R_idx]
	
	symbol = symbol_idx_map[sbymu_idx]
	ys = []
	xs = []
	
	if R_idx==0:
		color='0.7'
		zorder=1
	elif R_idx==2:
		color='r'
		zorder=2
	elif R_idx==3:
		color='g'
		zorder=2
	else:
		color='b'
		zorder=2
		
	line, = legend_axis.plot([-2],[-2],'.',label='$R/L\\mu=%g$' % rbymu,markersize=3,color=color,linewidth=1.5)
		
	for N in Ns:
		
		NS,NU,NR = run_simulation.calculate_scaled_parameters(sd,N,sbymu=sbymu)
		
		R = NR/N
		
		tc = N*syn_pis[(sbymu_idx,R_idx)][N]/2/NUn
		
		mean_tc = numpy.mean(tc)
		stderr_tc = numpy.std(tc)/numpy.sqrt(len(tc)*1.0)
		
		npfix = Npfixs[(sbymu_idx,R_idx)][N]
		mean_npfix = numpy.mean(npfix)
		stderr_npfix = numpy.std(npfix)/numpy.sqrt(len(npfix)*1.0)
		
		
		tc_axis.semilogx([N*sd/sbymu,N*sd/sbymu], [(mean_tc-2*stderr_tc)*sd/sbymu,(mean_tc+2*stderr_tc)*sd/sbymu],'-',color=color,zorder=zorder)
		tc_axis.loglog(N*sd/sbymu, tc.mean()*sd/sbymu,symbol,color=color,zorder=zorder,markersize=3)
		
		#pylab.semilogx([N*sd/sbymu,N*sd/sbymu], [(mean_npfix-2*stderr_npfix),(mean_npfix+2*stderr_npfix)],'-',color=color,zorder=zorder)
		pfix_axis.semilogy(mean_tc*sd, mean_npfix,symbol,color=color,zorder=zorder,markersize=3)
		print("mean_tc = %g, U*s=%g" % (mean_tc, NU/N*NS/N))

legend_axis.legend(loc='center left',frameon=False,numpoints=1,handletextpad=0.3,handlelength=0.8) #,fontsize=6)  


pylab.savefig('forward_time_simulations.pdf',bbox_inches='tight')
				
				
			
			 
		