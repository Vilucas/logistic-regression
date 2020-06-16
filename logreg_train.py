import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from src import DataFormat
from src.DataAnalyse import DataAnalyse

#get args
def     openDataFile():
    args = argparse.ArgumentParser()
    args.add_argument("data_file", help="data file needed")
    args.add_argument("--data_analyse", "-a", help="Plot the data analyse with sklearn", action="store_true")
    args.add_argument("--plot", "-p", help="Plot the training progress", action="store_true")
    args.add_argument("--verbose", "-v", help="increase output verbosity", action="store_true")
    return args.parse_args()

if __name__ == "__main__":    
    args = openDataFile()

    #try open csv file
    try:
        df = pd.read_csv(args.data_file)
    except:
        print("need data file (csv extension)")
        exit(1)

    #drop useless, isolate output
    format = DataFormat.DataFormat(args)
    df = df.dropna()
    df["Hogwarts House"] = df['Hogwarts House'].replace({"Gryffindor": 0, "Hufflepuff" : 1, "Slytherin" : 2, "Ravenclaw" : 3})
    df = df.drop('Index', axis = 1)
    DataAnalyse(df, args) #if -a
    houses = df["Hogwarts House"]
    df = df.select_dtypes([int, float])[['Astronomy', 'Charms', 'Herbology', 'Defense Against the Dark Arts', 'Herbology']]

    #switch to numpy array to manipulate data easily
    df_numpy = df.to_numpy()

    #normalization
    df_normalize = format.normalization(df)

    print(df_normalize.head(5))