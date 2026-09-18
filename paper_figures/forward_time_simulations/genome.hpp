#ifndef GENOME_HPP
#define GENOME_HPP

#include<vector>
#include "object_pool.hpp"

class Mutation{
    public:
        int location;
        //later on we will also have
        //double fitness effect
};

//inline bool operator<(Mutation const & m, const double p){ return m.location < p; }
//inline bool operator<(const double p, Mutation const & m){ return p < m.location; }
inline bool operator<(Mutation const & m1, Mutation const & m2){ return m1.location < m2.location; } 
//inline bool operator<=(Mutation const & m, const double p){ return m.location <= p; }
//inline bool operator<=(const double p, Mutation const & m){ return p <= m.location; }
inline bool operator<=(Mutation const & m1, Mutation const & m2){ return m1.location <= m2.location; }  
inline bool operator==(Mutation const & m1, Mutation const & m2){ return m1.location == m2.location; }
inline bool operator!=(Mutation const & m1, Mutation const & m2){ return m1.location != m2.location; }

typedef std::vector<Mutation> MutationList;
typedef SharedObjectPool<MutationList> GenomePool;

class Genome{
    public:

        // raw params (save to access later)
        int N;
        double Usyn;
        double Unon;
        double s;
        double R;
        double lfrac;
        double lavg;
      
        // processed params
        double fsyn; // fraction of nonsynonymous mutations
        double W;
        double NU;
        double NR;
        double NUn;
        
        int L;
        
        int neutral_locus_location;   
        
        GenomePool genome_pool;

        Genome(int N, double Usyn, double Unon, double s, double R, double lfrac): N(N), Usyn(Usyn), Unon(Unon), s(s), R(R), lfrac(lfrac) {

            NU = N*(Unon+Usyn);
                        
            W = exp(s);
            
            fsyn = Usyn/(Usyn+Unon);
            
            NR = N*R;
            NUn = 1;

            L = 1000000000; // make it huge!
            // we're then going to do a weird thing, where mutations can lie on even sites, and breakpoints can only occur on odd sites. Then you don't have to worry about something weird happening? 
            
            draw_site = create_random_int(0,L-1);

            lavg = lfrac * L;
            

            neutral_locus_location = -1; // was 0.5*L 
            // ensures that it doesn't get recombined
            
            genome_pool = GenomePool(4*N+4, MutationList());

            is_synonymous_mutation = create_bernoulli(fsyn);
            
            recombined = create_bernoulli(R);
            draw_num_mutations = create_poisson(Unon+Usyn);
            draw_num_neutral_mutations = create_poisson(NUn/N);

            draw_population_num_mutations = create_poisson(NU);
            draw_population_num_recombinants = create_poisson(NR);
            draw_population_num_neutral_mutations = create_poisson(NUn);
            
            draw_fragment_length = create_geometric(1.0/lavg); // we multiply by two because we add to both sides of midpoint

        };

        decltype(create_bernoulli(Usyn)) is_synonymous_mutation;
        
        decltype(create_random_int()) draw_site;
        
        decltype(create_bernoulli(R)) recombined;
        decltype(create_poisson(Unon)) draw_num_mutations;
        decltype(create_poisson(Usyn)) draw_num_neutral_mutations;

        decltype(create_poisson(NR)) draw_population_num_recombinants;
        decltype(create_poisson(NU)) draw_population_num_mutations;
        decltype(create_poisson(NUn)) draw_population_num_neutral_mutations;

        decltype(create_geometric(1/lavg)) draw_fragment_length;

        std::pair<int,int> draw_interval(Random & random){
            
            // first draw midpoint
            auto midpoint = draw_site(random);
            auto l = draw_fragment_length(random);
            
            auto start = midpoint;
            auto end = midpoint;
            
            if(l%2==0){
                auto dl = l/2;
                start = midpoint-dl+1;
                end = midpoint+dl;
            }
            else{
                auto dl = (l-1)/2;
                start = midpoint-dl;
                end = midpoint+dl;
            }
            
            start = (start >= 0) ? start : 0;
            end = (end < L) ? end : L-1;
            
            return std::pair<int,int>{start,end};
        }

        Mutation draw_mutation(Random & random){ 
            return Mutation{draw_site(random)}; 
        };
        
        double get_fitness_effect(Mutation & mutation){
        	return W; 
        }
};

#endif
