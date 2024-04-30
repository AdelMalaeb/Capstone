def Outliers_detection(col):
    '''
    Argument: Takes a column and performs the calaculations
    
    Calculates the Upper and lower limit, and filters all observations above the upper limit
    and below the lower limit
    
    '''
    

    #Define the 25th and the 75th percentile
    percentile25 = col.quantile(0.25)
    percentile75 = col.quantile(0.75)

    #Calculate the inter quantile range
    iqr = percentile75 - percentile25

    #Define the upper and lower bounds
    Upper_limit = percentile75 + 1.5 * iqr
    Lower_limit = percentile25 - 1.5 * iqr

    print("The upper limit is:", Upper_limit)
    print("The lower limit is:", Lower_limit)

    #Identify outliers
    outliers = df1[(col > Upper_limit) | (col < Lower_limit)]

    print("Number of outliers is:", len(outliers))
