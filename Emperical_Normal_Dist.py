import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
    
def emperical_normal (df):
    '''
    Arguments: Takes a dataframe as input
    Returns a dataframe that computes the intervals of the emperical normal distribution
    
    '''
  
    num = [1,2,3]
    emperical_df = pd.DataFrame()
    
    for i in num:
    
        mean = df.mean()
        std = df.std()
    
        lower_limit = mean - i * std
        upper_limit = mean + i * std
    
        emperical_dist = ((df >= lower_limit) & (df <= upper_limit)).mean()*100
    
        emperical_dict = {"Mean": mean,
                 "Std": std,
                 "lower_limit": lower_limit,
                 "upper_limit": upper_limit,
                 "Distribution percentage %":emperical_dist}
        
        
        index = [f'Interval {68 if i==1 else 95 if i==2 else 99.7}'"%".format(i)]  #Using list of comprehension
    
            
        emperical_df = emperical_df.append(pd.DataFrame(emperical_dict, index=index))
        
 
        
    return emperical_df
