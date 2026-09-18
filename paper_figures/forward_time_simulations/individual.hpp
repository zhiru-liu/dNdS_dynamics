#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include <vector>
#include <memory>
#include <algorithm>
#include "stats.hpp"
#include "genome.hpp"

class Individual{
   public:
       double location; // used for intrusive find while drawing parents
       double fitness; // used for memoization

       decltype(GenomePool().allocate()) nonsynonymous_mutations;
       decltype(GenomePool().allocate()) synonymous_mutations;
              
       std::shared_ptr<MutationList> neutral_locus;
       
       Individual() {};
       Individual(Genome & reference_genome): nonsynonymous_mutations(reference_genome.genome_pool.allocate()), synonymous_mutations(reference_genome.genome_pool.allocate()), neutral_locus(std::make_shared<MutationList>()) {};       

       void mutate_coding_region(Random & r, Genome & reference_genome);
       void add_mutation(Genome & reference_genome, decltype(nonsynonymous_mutations) & mutations, Mutation & mutation);
       void mutate_neutral_locus(Random & r, Genome & reference_genome);

       void recombine(Random & random, Genome & reference_genome, Individual & donor);
       void recombine_mutations(Genome & reference_genome, decltype(nonsynonymous_mutations) & recipient_mutations, decltype(nonsynonymous_mutations) & donor_mutations, int fragment_start, int fragment_end);
       void recalculate_fitness(Genome & reference_genome){ fitness = pow(reference_genome.W, nonsynonymous_mutations->size()); };

       bool operator<(const double p) const { return location < p; };        
};

inline void Individual::mutate_coding_region(Random & random, Genome & reference_genome){
    
    auto new_mutation = reference_genome.draw_mutation(random);
    
    if(reference_genome.is_synonymous_mutation(random)){
        add_mutation(reference_genome, synonymous_mutations, new_mutation);
    }
    else{
        add_mutation(reference_genome, nonsynonymous_mutations, new_mutation);
        recalculate_fitness(reference_genome);
    }
}
    
inline void Individual::add_mutation(Genome & reference_genome, decltype(nonsynonymous_mutations) & mutations, Mutation & new_mutation){
    
    auto new_mutations = reference_genome.genome_pool.allocate();
    auto insertion_point = std::upper_bound(mutations->begin(),mutations->end(),new_mutation);
    new_mutations->assign(mutations->begin(), insertion_point);
    new_mutations->push_back(new_mutation);
    if(mutations->end() != insertion_point)
        new_mutations->insert(new_mutations->end(), insertion_point, mutations->end());

    mutations.swap(new_mutations);
}

inline void Individual::mutate_neutral_locus(Random & random, Genome & reference_genome){
    std::make_shared<MutationList>(*neutral_locus).swap(neutral_locus);
    neutral_locus->push_back(reference_genome.draw_mutation(random));   
}

inline void Individual::recombine(Random & random, Genome & reference_genome, Individual & donor){
    auto interval = reference_genome.draw_interval(random);
    auto fragment_start = interval.first; // start of transferred fragment (inclusive)
    auto fragment_end = interval.second; // end of transferred fragment (inclusive)
    
    recombine_mutations(reference_genome, nonsynonymous_mutations, donor.nonsynonymous_mutations, fragment_start, fragment_end);
    recalculate_fitness(reference_genome);

    recombine_mutations(reference_genome, synonymous_mutations, donor.synonymous_mutations, fragment_start, fragment_end);

    // do neutral locus
    if((fragment_start <= reference_genome.neutral_locus_location) and (reference_genome.neutral_locus_location <= fragment_end))
        neutral_locus = donor.neutral_locus;
}

inline void Individual::recombine_mutations(Genome & reference_genome, decltype(nonsynonymous_mutations) & recipient_mutations, decltype(nonsynonymous_mutations) & donor_mutations, int fragment_start, int fragment_end){

    auto donor_start = donor_mutations->begin();
    auto donor_end = donor_mutations->end();
    
    auto donor_fragment_start = std::lower_bound(donor_start, donor_end, Mutation{fragment_start});
    auto donor_fragment_end = std::upper_bound(donor_start, donor_end, Mutation{fragment_end});
    
    // if there is something to transfer!
    auto recipient_start = recipient_mutations->begin();
    auto recipient_end = recipient_mutations->end();
        
    auto recipient_fragment_start = std::lower_bound(recipient_start, recipient_end, Mutation{fragment_start});
    auto recipient_fragment_end = std::upper_bound(recipient_start, recipient_end, Mutation{fragment_end});
    
    bool donor_fragment_nonempty = ((donor_fragment_start!=donor_end) && (donor_fragment_start!=donor_fragment_end));
    
    bool recipient_fragment_nonempty = ((recipient_fragment_start!=recipient_end) && (recipient_fragment_start!=recipient_fragment_end));
    
    if(recipient_fragment_nonempty || donor_fragment_nonempty){
           
        // do coding region
        auto new_mutations = reference_genome.genome_pool.allocate();
        new_mutations->clear();
        
        if(recipient_start != recipient_fragment_start){
            new_mutations->insert(new_mutations->end(),recipient_start, recipient_fragment_start);
        }  
        
        // already tested that there is something to transfer
        new_mutations->insert(new_mutations->end(), donor_fragment_start, donor_fragment_end);
         
        if(recipient_fragment_end != recipient_end){
            // something to keep after transfer
            new_mutations->insert(new_mutations->end(), recipient_fragment_end, recipient_end);
        }
        
        recipient_mutations.swap(new_mutations);
       
    }
}

typedef std::vector<Individual> Population;
    
inline Population draw_random_sample(Random & random, Population const & population, int n){
    Population sample(n,population[0]);
    auto draw_random_index = create_random_int(0,population.size()-1);
    auto draw_random_individual = [&]()->Individual const &{ return population[draw_random_index(random)]; }; 
    std::generate(sample.begin(),sample.end(),draw_random_individual);
    return sample;
}

#endif
