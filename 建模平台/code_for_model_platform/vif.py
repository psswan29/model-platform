
import pandas as pd

def calulate_vif(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X[X.shape[1]]=1
    #vif
    vif=[]
    for i in range(X.shape[1]-1):
        vif.append(variance_inflation_factor(X.values,i))
    #result_out
    yy=pd.DataFrame(X.columns[:-1,])
    yy.rename(columns={0:"var_name"},inplace=True)
    yy["vif"]=vif
    print(yy)

if __name__ == '__main__':
    df = pd.DataFrame([[15.9, 16.4, 19, 19.1, 18.8, 20.4, 22.7, 26.5, 28.1, 27.6, 26.3]
                          , [149.3, 161.2, 171.5, 175.5, 180.8, 190.7, 202.1, 212.1, 226.1, 231.9, 239]
                          , [4.2, 4.1, 3.1, 3.1, 1.1, 2.2, 2.1, 5.6, 5, 5.1, 0.7]
                          , [108.1, 114.8, 123.2, 126.9, 132.1, 137.7, 146, 154.1, 162.3, 164.3, 167.6]]).T
    columns = ["var1", "var2", "var3", "var4"]
    df.columns = columns
    calulate_vif(df)