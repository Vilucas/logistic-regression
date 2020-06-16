import numpy as np

class DataFormat:
    def __init__(self, args):
        self.plot = args.plot
        self.verbose = args.verbose
    
    def normalization(self, df):
       return (df - np.min(df, axis=0)) / (np.max(df, axis=0) - np.min(df, axis=0))