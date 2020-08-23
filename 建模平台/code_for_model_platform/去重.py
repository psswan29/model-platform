#去重
def drop_dup(data1,subset=None,keep='first'):
    '''
    data1:数据集
    subset：根据哪几列进行去重,默认None即根据所有列去重。
    keep:重复数据保留第一条-'first',最后一条-'last'
    '''
    df = data1.drop_duplicates(subset=subset,keep=keep)
    return df

