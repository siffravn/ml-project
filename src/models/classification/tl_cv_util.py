# -*- coding: utf-8 -*-

import numpy as np
import ann

def find_optimal_model(parameters, val_errors, weight):
    
    # Compute model generalization error for each model s
    gen_errors = np.zeros(shape=(len(parameters)))
    for index in range(len(parameters)):
        p = parameters[index]
        
        errors = val_errors[:,index]
        
        gen_errors[index] =sum(weight*errors)
        
        
    # Find and select optimal model 
    index = gen_errors.argmin()
    optimal_parameter = parameters[index]
        
    return optimal_parameter

    # gen_errors = np.zeros(shape=(len(hidden_units)))

    # for index in range(len(hidden_units)):
    #     h = hidden_units[index]
        
    #     errors = ann_val_errors[i,:,index]
        
    #     gen_error = (len(test_index2)/len(train_index1))*errors
    #     gen_errors[index] = gen_error.sum()
    
    
    # # Find and select optimal model 
    # index = gen_errors.argmin()
    # optimal_hidden_units = hidden_units[index]