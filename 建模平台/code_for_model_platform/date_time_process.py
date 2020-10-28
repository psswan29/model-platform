import datetime
import pandas as pd
import time

# 本身pandas已经提供了许多方法处理时间日期问题，但是有时需要我们手动处理这类问题
current_date = datetime.datetime.today()


def parse_2_datetime(data_1, var_name, format='%m/%d/%Y',freq='month'):
    """

    :param data_1:
    :param var_name:
    :param format:
    :param freq:
    :return:
    """
    if freq == 'month':
        return  ((datetime.datetime(2020, 10, 28) - pd.to_datetime(data_1[var_name],
                                                       format=format)).dt.days // 30).astype(int)
    else:
        return ((datetime.datetime(2020, 10, 28) - pd.to_datetime(data_1[var_name],
                                                format=format)).dt.days).astype(int)




