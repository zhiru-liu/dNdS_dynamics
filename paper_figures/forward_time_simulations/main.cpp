#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <math.h>
#include "stats.hpp"
#include "genome.hpp"
#include "individual.hpp"

class Measurement {
    public:
        int time;
        double avg_fitness;
        double fixed_fitness;
        Population sample;
        //std::vector<int> synonymous_difference_matrix;
        //std::vector<int> nonsynonymous_difference_matrix;
};

class Results {
    public:
        int equilibrium_time;
        double equilibrium_avg_fitness;
        double equilibrium_fixed_fitness;
        MutationList fixed_mutations;
        std::vector<Measurement> measurements;
        
};


Results evolve_population(Random & random, double floatN, Genome & genome, int t_max, int n);
double remove_fixed_synonymous_mutations(Population & population, Genome & genome);
double remove_fixed_nonsynonymous_mutations(Population & population, Genome & genome);


int main(int argc, char * argv[]){
    if(argc < 10){
        std::cout << "usage: " << argv[0] << " n_replicates t_max n N Usyn Unon s R lfrac" << std::endl;
        return 1;
    }
    else{
        unsigned long num_replicates = atol(argv[1]); // num replicate populations to evolve
        unsigned long t_max = atol(argv[2]); // max number of generations to run for
        unsigned long n = atol(argv[3]); // sample size for looking at genetic diversity
        double N = atof(argv[4]); // population size
        double Usyn = atof(argv[5]); // total mutation rate for neutral (synonymous) mutations
        double Unon = atof(argv[6]); // total mutation rate for selected (nonsynonymous) mutations
        double s = atof(argv[7]);
        double R = atof(argv[8]);
        double lfrac = atof(argv[9]);

        //Random random = create_random(42);
        Random random = create_random();
        Genome genome(N,Usyn,Unon,s,R,lfrac);
        
        // print header
        std::cout << "Individual (pop;t;fixed_fitness;idx), Absolute Fitness (x-x0bar), Synonymous mutations, Nonsynonymous mutations" << std::endl; 
        for(int replicate_idx=0;replicate_idx<num_replicates;++replicate_idx){
            Results results = evolve_population(random,N,genome,t_max,n);
            if(results.measurements.empty()){
                std::cerr << "Something weird happened: no measurements!" << std::endl;
            }
            else{

                for(auto & measurement : results.measurements){
                
                	int dt = measurement.time-results.equilibrium_time;
                
                    for(int i=0,imax=measurement.sample.size();i<imax;++i){
                        std::cout << replicate_idx+1 << ";" << dt << ";" << std::log(measurement.fixed_fitness/results.equilibrium_fixed_fitness) << ";" << i+1 << ", " << std::log(measurement.sample[i].fitness*measurement.fixed_fitness/results.equilibrium_avg_fitness) << ", ";
                        // first print synonymous mutations
                        for(auto mutation_ptr=measurement.sample[i].synonymous_mutations->begin(), end=measurement.sample[i].synonymous_mutations->end();mutation_ptr!=end;++mutation_ptr){
                            std::cout << mutation_ptr->location*1.0/genome.L << " ";
                        }
                        std::cout << ", ";
                        for(auto mutation_ptr=measurement.sample[i].nonsynonymous_mutations->begin(), end=measurement.sample[i].nonsynonymous_mutations->end();mutation_ptr!=end;++mutation_ptr){
                            std::cout << mutation_ptr->location*1.0/genome.L << " ";
                        }
                        std::cout << std::endl;
                    }
                    
                }
                
            }
        }
        
    }
}

Results evolve_population(Random & random, double floatN, Genome & genome, int t_max, int n){

    Results results;
    int N = floatN;
    auto draw_index = create_random_int(0,N-1);

    Population population(N, Individual(genome)); // the current population of individuals
    Population new_population(N, population.front()); // used for constructing the new generation     
    Population sample(n, population.front()); // for sampling diversity  
    
    // initialize population
    double total_fitness = 0;
    for(auto & individual : population){
        individual.mutate_neutral_locus(random, genome);
        individual.fitness = 1.0;
        total_fitness += individual.fitness;
        individual.location = total_fitness;
    }

    bool in_equilibrium = false;
    bool first_in_equilibrium = false;
    int deltat = 0;
    double fixed_fitness = 1.0;
    //std::cerr << t_max << std::endl;
    for(int t=0;t<t_max;++t){
        
        // calculate total fitness and "location" of each individual (for sampling)
        total_fitness = 0;        
        for(auto & individual : population){
            total_fitness += individual.fitness;
            individual.location = total_fitness;
        }
        
        auto draw_parent = [&,total_fitness](Random & r)->Individual & { return *std::lower_bound(population.begin(),population.end(),total_fitness*sample_uniform(r)); };
        
        for(auto & individual : new_population){
            individual = draw_parent(random);
            //individual.mutate_marker_loci(random, genome);
        }

        // do recombination 
        for(int i=0,num_recombinants = genome.draw_population_num_recombinants(random);i<num_recombinants;++i){
            new_population[draw_index(random)].recombine(random, genome, draw_parent(random)); 
        }

        // add genomic mutations
        for(int i=0,num_mutations = genome.draw_population_num_mutations(random);i<num_mutations;++i){
            new_population[draw_index(random)].mutate_coding_region(random, genome); 
        }

        // add neutral loci mutations
        for(int i=0,num_mutations = genome.draw_population_num_neutral_mutations(random);i<num_mutations;++i){
            new_population[draw_index(random)].mutate_neutral_locus(random, genome); 
        }

        total_fitness = 0;        
        for(auto & individual : new_population){
            total_fitness += individual.fitness;
            individual.location = total_fitness;
        }
        std::swap(population,new_population);

		// check to see if a mutation has fixed at the neutral locus
        // (used for determining whether we are in "equilibrium")
        if(!population.front().neutral_locus->empty() ){ 
            // a mutation can only fix if the first individual has it
            auto first_mutation = population.front().neutral_locus->front();
            auto same_first_mutation = [first_mutation](Individual const & individual){return (!individual.neutral_locus->empty()) && (individual.neutral_locus->front() == first_mutation);};
 
            if(std::all_of(++population.begin(),population.end(),same_first_mutation)){
                for(auto & individual : population){
                    if(same_first_mutation(individual)) 
                        individual.neutral_locus->erase(individual.neutral_locus->begin());
                }
                results.fixed_mutations.push_back(first_mutation);
                
                // if we aren't in equilibrium yet, then this means
                // we have just reached equilibrium
                if(!in_equilibrium){
                	first_in_equilibrium = true;
                }
            }
        }
 
		// for computational efficiency, remove fixed mutations
        if(t % 100 == 0 || first_in_equilibrium || t==(t_max-1)){
            remove_fixed_synonymous_mutations(population, genome);
            double fitness_increment = remove_fixed_nonsynonymous_mutations(population, genome);
            fixed_fitness *= fitness_increment;
            total_fitness /= fitness_increment;
            //std::cerr << "t = " << t << ", " << fitness_increment << ", " << total_fitness/floatN << ", " << t_max << std::endl; // print status
            
        }
        
        // if we just reached equilibrium, do some things
        if(first_in_equilibrium){
            in_equilibrium = true;
            first_in_equilibrium = false;
            results.equilibrium_time = t;
            results.equilibrium_avg_fitness = total_fitness/floatN*fixed_fitness;
            results.equilibrium_fixed_fitness = fixed_fitness;
            deltat = t;
            t_max = 10*t; // re-adjust tmax based on local equilibration time
            std::cerr << "Equilibrated after " << t << " generations; " << results.equilibrium_fixed_fitness << " " << results.equilibrium_avg_fitness << std::endl;
        }

        // record stuff if we're at the end...
        if(t==t_max-1){ 
        	//std::cerr << "Recording..." << std::endl;
            for(auto & individual : sample){
                individual = population[draw_index(random)];
            }
            results.measurements.push_back(Measurement{t, total_fitness/floatN*fixed_fitness,fixed_fitness, sample});
        }


        


    }
    return results;
}


double remove_fixed_synonymous_mutations(Population & population, Genome & genome){
    
    MutationList fixed_mutations(population.front().synonymous_mutations->begin(),population.front().synonymous_mutations->end());
    MutationList new_fixed_mutations = fixed_mutations;

    //std::cout << fixed_mutations.size() << std::endl;

    for(auto & individual : population){
        auto new_end = std::set_intersection(fixed_mutations.begin(),fixed_mutations.end(), individual.synonymous_mutations->begin(),individual.synonymous_mutations->end(), new_fixed_mutations.begin());
        new_fixed_mutations.resize(new_end-new_fixed_mutations.begin());
        std::swap(fixed_mutations, new_fixed_mutations);
        if(fixed_mutations.empty()) break;
    }
    if(!fixed_mutations.empty()){
        //std::cout << "Removing genomic mutations!" << std::endl;
            for(auto & mutation : fixed_mutations){
                for(auto & individual : population){
                    auto mutation_ptr = std::lower_bound(individual.synonymous_mutations->begin(), individual.synonymous_mutations->end(), mutation);
                    if(mutation_ptr != individual.synonymous_mutations->end() && *mutation_ptr == mutation){
                        individual.synonymous_mutations->erase(mutation_ptr);
                    }
                }
            }
    }
    
    return 1.0;
}

double remove_fixed_nonsynonymous_mutations(Population & population, Genome & genome){
    
    MutationList fixed_mutations(population.front().nonsynonymous_mutations->begin(),population.front().nonsynonymous_mutations->end());
    MutationList new_fixed_mutations = fixed_mutations;

    //std::cout << fixed_mutations.size() << std::endl;

    for(auto & individual : population){
        auto new_end = std::set_intersection(fixed_mutations.begin(),fixed_mutations.end(), individual.nonsynonymous_mutations->begin(),individual.nonsynonymous_mutations->end(), new_fixed_mutations.begin());
        new_fixed_mutations.resize(new_end-new_fixed_mutations.begin());
        std::swap(fixed_mutations, new_fixed_mutations);
        if(fixed_mutations.empty()) break;
    }
    
    double fitness_increment = 1.0;
    if(!fixed_mutations.empty()){
        //std::cout << "Removing genomic mutations!" << std::endl;
        for(auto & mutation : fixed_mutations){
            
            fitness_increment *= genome.get_fitness_effect(mutation);
            
            for(auto & individual : population){
                auto mutation_ptr = std::lower_bound(individual.nonsynonymous_mutations->begin(), individual.nonsynonymous_mutations->end(), mutation);
                if(mutation_ptr != individual.nonsynonymous_mutations->end() && *mutation_ptr == mutation){
                    individual.nonsynonymous_mutations->erase(mutation_ptr);
                }
            }
        }
        
        // recalculate fitnesses since we removed some things
        for(auto & individual : population){
            individual.recalculate_fitness(genome);
        }
            
    }
    
    return fitness_increment;
}







