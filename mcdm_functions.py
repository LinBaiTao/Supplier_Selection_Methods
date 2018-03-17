# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 07:53:25 2018

@author: Nickolas K. Freeman, Ph.D.
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

#######################################################################################
def Normalize_Weights(weights_array,norm_type = 'divide_by_sum'):
    """Normalizes a provided weight array so that the sum equals 1

    Parameters
    ----------
    weights_array : a numpy array containing the raw weights
    
    norm_type : a string specifying the type of normalization to perform
        - 'divide_by_max' divides all values by the maximum value
        - 'divide_by_sum' divides all values by the sum of the values

    Yields
    ------
    temp_weights_array: a copy of the passed weights array with the normalizations performed

    Examples
    --------
    >>> import numpy as np
    >>> import mcdm_functions as mcfunc
    >>> criteria_weights = np.array([2,4,6,7,9])
    >>> temp = mcfunc.Normalize_Weights(criteria_weights,'divide_by_max')
    >>> print(temp)
    
        [ 0.22222222  0.44444444  0.66666667  0.77777778  1.        ]
    """   
    
    temp_weights_array = weights_array.copy()
    
    if norm_type is 'divide_by_max':
        temp_weights_array = temp_weights_array/temp_weights_array.max()

    elif norm_type is 'divide_by_sum':
        temp_weights_array = temp_weights_array/temp_weights_array.sum()
        
    else:
        print('You did not enter a valid type, so no changes were made')
    
    return temp_weights_array

        
        
        
###########################################################################################
def Convert_Higher_is_Better(df, columns, conversion_type = 'absolute'):
    """Converts scores given in a "lower is better" format to a "higher is better" format

    Parameters
    ----------
    df : a pandas DataFrame object that contains the specified columns
    
    columns: a list object that includes the columns to normalize
    
    conversion_type : a string specifying the type of conversion to perform
        - 'absolute' converts based on absolute scale
        - 'relative' converts based on relative scale
        
    Yields
    ------
    temp_df: a copy of the passed dataframe with the conversions performed
    """
    
    
    temp_df = df.copy()
        
    for column in columns:
        if conversion_type is 'absolute':
            new_column = column+' (absolute HIB)'
            max_entry = temp_df[column].max()
            temp_df[new_column] = max_entry - temp_df[column]
    
        elif conversion_type is 'relative':
            new_column = column+' (relative HIB)'
            min_entry = temp_df[column].min()
            max_entry = temp_df[column].max()
            temp_df[new_column] = (max_entry - temp_df[column])/(max_entry-min_entry)

        else:
            print('You did not enter a valid type, so no changes were made')        
        
    return temp_df    



###########################################################################################
def Normalize_Column_Scores(df, columns, norm_type = 'divide_by_max'):
    """Normalizes scores for specified columns in a pandas dataframe

    Parameters
    ----------
    df : a pandas DataFrame object that contains the specified columns
    
    columns: a list object that includes the columns to normalize
    
    norm_type : a string specifying the type of normalization to perform
        - 'divide_by_max' divides all values by the maximum value
        - 'range_norm' divides all values (+ the min) by the range of values in the column
        - 'z_norm' computes a z-score based on the mean and standard deviation of values
        - 'divide_by_sum' divides all values by the sum of the values
        - 'vector' dives all values by the square root of the sum of the squares of all values

    Yields
    ------
    temp_df: a copy of the passed dataframe with the normalizations performed

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import mcdm_functions as mcfunc

    >>> data_dict = {'Product': ['A', 'B', 'C', 'D'],
                 'Product Advantage': [13.1,13.2,12.2,13.2],
                 'Strategic Alignment': [9.8,8.2,10.0,9.6],
                 'Technical Feasibility': [20.0,18.7,18.5,17.1],
                 'Market Attractiveness': [15.5,12.3,13.1,13.1]}
    >>> score_data = pd.DataFrame(data_dict)
    >>> score_data = score_data.set_index('Product')
    >>> print(score_data)
    
            Market Attractiveness  Product Advantage  Strategic Alignment  \
    Product                                                                  
    A                         15.5               13.1                  9.8   
    B                         12.3               13.2                  8.2   
    C                         13.1               12.2                 10.0   
    D                         13.1               13.2                  9.6   

             Technical Feasibility  
    Product                         
    A                         20.0  
    B                         18.7  
    C                         18.5  
    D                         17.1  
    
    
    >>> columns = ['Market Attractiveness','Product Advantage']
    >>> temp = mcfunc.Normalize_Column_Scores(score_data,columns)
    >>> print(temp)
    
             Market Attractiveness  Product Advantage  Strategic Alignment  \
    Product                                                                  
    A                     1.000000               13.1                  9.8   
    B                     0.793548               13.2                  8.2   
    C                     0.845161               12.2                 10.0   
    D                     0.845161               13.2                  9.6   

             Technical Feasibility  
    Product                         
    A                         20.0  
    B                         18.7  
    C                         18.5  
    D                         17.1  
    """   
    
    temp_df = df.copy()
    
    for column in columns:
        if norm_type is 'divide_by_max':
            max_entry = temp_df[column].max()
            temp_df[column] = temp_df[column]/max_entry
        
        elif norm_type is 'range_norm':
            min_entry = temp_df[column].min()
            max_entry = temp_df[column].max()
            temp_df[column] = (temp_df[column]-min_entry)/(max_entry - min_entry)
        
        elif norm_type is 'z_norm':
            mean = temp_df[column].mean()
            sd = temp_df[column].std()
            temp_df[column] = (temp_df[column]-mean)/sd
        
        elif norm_type is 'divide_by_sum':
            temp_df[column] = temp_df[column]/temp_df[column].sum()
            
        elif norm_type is 'vector':
            values = temp_df[column].values
            values_squared = values**2
            vector_norm = values/np.sqrt(np.sum(values_squared))
            temp_df[column] = vector_norm
        
        else:
            print('You did not enter a valid type, so no changes were made')
        
    return temp_df
            
            
###################################################################################
def Compute_Weighted_Sum_Score(df,criteria_list,weights_array):
    """Computes weighted sum score for specified columns

    Parameters
    ----------
    df : a pandas DataFrame object that contains the specified columns and scores
    
    columns: a list object that includes the columns to include in the score
    
    weights_array : an array containing the weights for each column
    
    Yields
    ------
    temp_df: a sorted copy of the passed dataframe with the score in a 
    new column named 'Weighted Score (Sum)'

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import mcdm_functions as mcfunc

    >>> data_dict = {'Product': ['A', 'B', 'C', 'D'],
                 'Product Advantage': [13.1,13.2,12.2,13.2],
                 'Strategic Alignment': [9.8,8.2,10.0,9.6],
                 'Technical Feasibility': [20.0,18.7,18.5,17.1],
                 'Market Attractiveness': [15.5,12.3,13.1,13.1]}
    >>> score_data = pd.DataFrame(data_dict)
    >>> score_data = score_data.set_index('Product')
    >>> print(score_data)
    
            Market Attractiveness  Product Advantage  Strategic Alignment  \
    Product                                                                  
    A                         15.5               13.1                  9.8   
    B                         12.3               13.2                  8.2   
    C                         13.1               12.2                 10.0   
    D                         13.1               13.2                  9.6   

             Technical Feasibility  
    Product                         
    A                         20.0  
    B                         18.7  
    C                         18.5  
    D                         17.1  
    
    >>> criteria = ['Market Attractiveness',
                    'Product Advantage',
                    'Strategic Alignment',
                    'Technical Feasibility']

    >>> criteria_weights = np.array([4,6,7,9])

    >>> temp = mcfunc.Compute_Weighted_Sum_Score(score_data,criteria,criteria_weights)
    >>> print(temp)
    
             Market Attractiveness  Product Advantage  Strategic Alignment  \
    Product                                                                  
    A                         15.5               13.1                  9.8   
    B                         12.3               13.2                  8.2   
    C                         13.1               12.2                 10.0   
    D                         13.1               13.2                  9.6   

             Technical Feasibility  Weighted Score (Sum)  
    Product                                               
    A                         20.0                 389.2  
    B                         18.7                 354.1  
    C                         18.5                 362.1  
    D                         17.1                 352.7  
       
    """   
    temp_df = df.copy()
    temp_weights = weights_array.copy()
    
    temp_df['Weighted Score (Sum)'] = 0
    for i in range(len(criteria_list)):
        current_criteria = criteria_list[i]
        current_weight = temp_weights[i]
        temp_df['Weighted Score (Sum)'] += current_weight*temp_df[current_criteria]
    
    temp_df.sort_values(by = 'Weighted Score (Sum)', ascending=False, inplace=True)
    return temp_df   



###################################################################################
def Compute_Weighted_Product_Score(df,criteria_list,weights_array):
    """Computes weighted product score for specified columns

    Parameters
    ----------
    df : a pandas DataFrame object that contains the specified columns and scores
    
    columns: a list object that includes the columns to include in the score
    
    weights_array : an array containing the weights for each column
    
    Yields
    ------
    temp_df: a sorted copy of the passed dataframe with the score in a 
    new column named 'Weighted Score (Product)'

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import mcdm_functions as mcfunc

    >>> data_dict = {'Product': ['A', 'B', 'C', 'D'],
                 'Product Advantage': [13.1,13.2,12.2,13.2],
                 'Strategic Alignment': [9.8,8.2,10.0,9.6],
                 'Technical Feasibility': [20.0,18.7,18.5,17.1],
                 'Market Attractiveness': [15.5,12.3,13.1,13.1]}
    >>> score_data = pd.DataFrame(data_dict)
    >>> score_data = score_data.set_index('Product')
    >>> print(score_data)
    
            Market Attractiveness  Product Advantage  Strategic Alignment  \
    Product                                                                  
    A                         15.5               13.1                  9.8   
    B                         12.3               13.2                  8.2   
    C                         13.1               12.2                 10.0   
    D                         13.1               13.2                  9.6   

             Technical Feasibility  
    Product                         
    A                         20.0  
    B                         18.7  
    C                         18.5  
    D                         17.1  
    
    >>> criteria = ['Market Attractiveness',
                    'Product Advantage',
                    'Strategic Alignment',
                    'Technical Feasibility']

    >>> criteria_weights = np.array([0.1, 0.2, 0.3, 0.4])

    >>> temp = mcfunc.Compute_Weighted_Product_Score(score_data,criteria,criteria_weights)
    >>> print(temp)
    
               Market Attractiveness  Product Advantage  Strategic Alignment  \
    Product                                                                  
    A                         15.5               13.1                  9.8   
    B                         12.3               13.2                  8.2   
    C                         13.1               12.2                 10.0   
    D                         13.1               13.2                  9.6   

             Technical Feasibility  Weighted Score (Product)  
    Product                                                   
    A                         20.0                 14.463295  
    B                         18.7                 13.061291  
    C                         18.5                 13.673125  
    D                         17.1                 13.296022   
       
    """   
    temp_df = df.copy()
    temp_weights = weights_array.copy()
    
    temp_df['Weighted Score (Product)'] = 1
    for i in range(len(criteria_list)):
        current_criteria = criteria_list[i]
        current_weight = temp_weights[i]
        temp_df['Weighted Score (Product)'] *= temp_df[current_criteria]**current_weight
    
    temp_df.sort_values(by = 'Weighted Score (Product)', ascending=False, inplace=True)
    return temp_df   



def Compute_TOPSIS_Score(df,criteria_list,weights_array):
    '''This function computes a TOPSIS score for suppliers based on the specified criteria and weights
    
    
    """Computes weighted product score for specified columns

    Parameters
    ----------
    df : a pandas DataFrame object that contains the specified columns and scores
    
    columns: a list object that includes the columns to include in the score
    
    weights_array : an array containing the weights for each column
    
    Yields
    ------
    temp_df: a sorted copy of the passed dataframe with the score in a 
    new column named 'TOPSIS Score'
    
    Note
    ----
    This TOPSIS implementation does not perform any standardization of the alernative scores or weights. 
    We do this to allow the user to define and experiment with various methods of regularization.
    
    '''
    
    temp_df = df.copy()
    temp_weights_array = weights_array.copy()
    
    #Step 1
    evaluation_matrix = temp_df[criteria_list].values

    #Step 2
    #squared_evaluation_matrix = evaluation_matrix**2
    #normalized_evaluation_matrix = evaluation_matrix/np.sqrt(np.sum(squared_evaluation_matrix,axis=0))
    normalized_evaluation_matrix = evaluation_matrix

    #Step 3
    #temp_weights_array = temp_weights_array/temp_weights_array.sum()
    weighted_matrix = normalized_evaluation_matrix * temp_weights_array

    #Step 4
    PIS = np.max(weighted_matrix, axis=0)
    NIS = np.min(weighted_matrix, axis=0)

    #Step 5
    intermediate = (weighted_matrix - PIS)**2
    Dev_Best = np.sqrt(intermediate.sum(axis = 1))

    intermediate = (weighted_matrix - NIS)**2
    Dev_Worst = np.sqrt(intermediate.sum(axis = 1))

    #Step 6
    Closeness = Dev_Worst/(Dev_Best+Dev_Worst)

    #Step 7
    temp_df['TOPSIS Score'] = Closeness.tolist()
    temp_df.sort_values(by='TOPSIS Score',ascending=False,inplace=True)

    return temp_df



##############################################################################
def Robust_Ranking(df, 
                   columns,
                   perturbations,
                   weights_array,
                   index_column,
                   weights_min=0,
                   weights_max= 9999999999,
                   perturbation_range = 0.05,
                   weight_norm_method ='divide_by_sum',
                   score_type = 'weighted_sum',
                   top_values = 5,
                   include_plot=False
                  ):
    """Performs a robustness check for defined ranking methods

    Parameters
    ----------
    df : a pandas DataFrame object that contains the specified columns
    
    columns: a list object that includes the columns to include in the score
    
    weights_array: an array containing the weights for each column
    
    index_column: a string specifying the column of alternative
    
    weights_min: The minimum aceptable value for weights
    
    weights_max: The maximum aceptable value for weights
    
    perturbation_range: The perturbation range to consider (expressed as a proportion of the mean value)
    
    weight_norm_method: A string specifying the method to use for normalization of perturbed weights
        - 'divide_by_max' divides all weights by the maximum weight
        - 'divide_by_sum' divides all weights by the sum of the weights
    
    score_type: A string specifying the type of score  calculation
        - 'weighted_sum' uses the weighted sum method
        - 'weighted_product' uses the weighted product method
        - 'topsis'
    
    top_values: specifies the number of alternatives to keep from each ranking (highest ranked scores kept)

    Yields
    ------
    sum_counts: dataframe indicating the proportion of times each alternative appears in top ranking
    """
    
    
    temp_perturbations = perturbations
    df_copy = df.copy()
    temp_columns_list = list(columns)
    temp_weights_array = weights_array.copy()
    
    weight_mean = temp_weights_array.mean()
    perturbation_amount = weight_mean*perturbation_range
            
    supplier_list = []
    count_column_name = ''
    proportion_column_name = ''
       
    np.random.seed(42)

    for i in range(temp_perturbations):
        perturbation_vector = np.random.uniform(low = -1.0*perturbation_amount,
                                                high = perturbation_amount,
                                                size=len(weights_array))

        perturbed_weights = temp_weights_array + perturbation_vector
        perturbed_weights = np.maximum(perturbed_weights,weights_min)
        perturbed_weights = np.minimum(perturbed_weights,weights_max)
        
        if weight_norm_method == 'divide_by_sum':
            perturbed_weights = Normalize_Weights(perturbed_weights,'divide_by_sum')
        
        elif weight_norm_method == 'divide_by_max':
            perturbed_weights = Normalize_Weights(perturbed_weights,'divide_by_max')
        
        if score_type == 'weighted_sum':
            count_column_name = 'Weighted Sum Count'
            proportion_column_name = 'Weighted Sum Proportion'
            temp_df = Compute_Weighted_Sum_Score(df_copy,
                                                 temp_columns_list,
                                                 perturbed_weights)
            temp_df = temp_df.nlargest(top_values, columns = 'Weighted Score (Sum)')
            temp_list = temp_df[index_column].values.tolist()
            supplier_list.append(temp_list)
            
        elif score_type == 'weighted_product':
            count_column_name = 'Weighted Product Count'
            proportion_column_name = 'Weighted Product Proportion'
            temp_df = Compute_Weighted_Product_Score(df_copy,
                                                 temp_columns_list,
                                                 perturbed_weights)
            temp_df = temp_df.nlargest(top_values, columns = 'Weighted Score (Product)')
            temp_list = temp_df[index_column].values.tolist()
            supplier_list.append(temp_list)
            
        elif score_type == 'topsis':
            count_column_name = 'TOPSIS Count'
            proportion_column_name = 'TOPSIS Proportion'
            temp_df = Compute_TOPSIS_Score(df_copy,
                                           temp_columns_list,
                                           perturbed_weights)
            temp_df = temp_df.nlargest(top_values, columns = 'TOPSIS Score')
            temp_list = temp_df[index_column].values.tolist()
            supplier_list.append(temp_list)
           
    
    sum_counts = Counter(x for sublist in supplier_list for x in sublist)
    sum_counts = pd.DataFrame.from_dict(sum_counts,orient='index').reset_index()
    sum_counts = sum_counts.rename(columns={'index':index_column,0:count_column_name})
    sum_counts.sort_values(by=count_column_name, ascending=False,inplace=True)
    sum_counts[proportion_column_name] = sum_counts[count_column_name]/(temp_perturbations)
    
    
    if include_plot is True:
        fig, ax = plt.subplots(1,1,figsize = (8,5))

        sum_counts.plot.bar(x=index_column,
                            y=proportion_column_name,
                            ax = ax,
                            color = 'blue',
                            alpha = 0.5,
                            edgecolor = 'k')

        ax.set_title(proportion_column_name,fontsize = 20)
        ax.set_xlabel('Alternative',fontsize = 20)
        ax.set_ylabel('Proportion',fontsize = 20)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)

        ax.legend_.remove()
        plt.show()
    
    return sum_counts