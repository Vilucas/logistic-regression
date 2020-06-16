from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import matplotlib.pyplot as plt

#understand which features are the most relevant to get the best prediction %tage
def DataAnalyse(df, args):
    if (args.plot == False):#dwedew
        return
    
    #purge non numerical values
    df = df.select_dtypes([int, float])
    
    #ExtraTrees can't hanle floats ?? Weird
    X = round(df.iloc[:,0:20]) #taking all features
    y = round(df.iloc[:,0]) #y is houses
    
    model = ExtraTreesClassifier() 
    model.fit(X,y)
    
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot.barh()

    plt.title("Relevance of features toward the output")
    plt.show()