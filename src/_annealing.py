import os

from _mutagenesis import control_edit, gen_edit
from _fitness import evaluate_fitness, evaluate_acceptance

# Run simulated annealing - one chain
class run_simulated_annealing:

    def __init__(self, predict_func,
                       init_bundle,
                       action_probs={'SUB' : 0.7, 
                                     'INS' : 0.15, 
                                     'DEL' : 0.05, 
                                     'SWITCH_POS' : 0.05, 
                                     'INS_WT' : 0.025, 
                                     'DEL_WT' : 0.025},
                       init_fitness=1.):

        self.predict_func=predict_func            
        self.edit_history=init_bundle
        self.action_probs=action_probs
        self.fitness = init_fitness
        self.score = None
    
    # Function to simulate annealing
    def temp_annealing(self, curr_iter, n_iters=1000,
                       temperature_range=(0.001,0.0001),
                       exp_scale=1./0.7): 
        
        t_min = temperature_range[-1]
        t_init = temperature_range[0]

        self.temperature = t_init * (t_min/t_init)**(min(float(curr_iter / n_iters) * exp_scale, 1.0))

    def iterate(self, clip_prob=0.9, objective='max', 
                max_edited_bp=10, max_overwritten_wt_bp=3, 
                insert_offset_range=(-2, 2), **kwargs):

        self.clip_prob=clip_prob

        new_action_probs = control_edit(self.edit_history, 
                                        action_probs=self.action_probs,
                                        max_edited_bp=max_edited_bp, 
                                        max_overwritten_wt_bp=max_overwritten_wt_bp, 
                                        insert_offset_range=insert_offset_range)
        
        edit_bundle = gen_edit(self.edit_history, 
                               action_probs=new_action_probs,
                               max_edited_bp=max_edited_bp, 
                               max_overwritten_wt_bp=max_overwritten_wt_bp, 
                               insert_offset_range=insert_offset_range)

        fitness_new, score_new = evaluate_fitness(edit_bundle, self.predict_func, 
                                                  objective=objective, **kwargs)

        accepts = evaluate_acceptance(fitness_new, self.fitness, 
                                      temperature=self.temperature, 
                                      clip_prob=self.clip_prob)
        
        if accepts:
            self.edit_history = edit_bundle
            self.action_probs = new_action_probs
            self.fitness = fitness_new
            self.score = score_new