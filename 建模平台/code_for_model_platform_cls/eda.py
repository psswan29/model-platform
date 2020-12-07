import numpy as np
import pandas as pd
from pyecharts.charts import Bar, Line
import pyecharts.options as opts
from pyecharts.globals import ThemeType
import qgrid
import pyecharts


class EDA(object):

    def __init__(self, X, y, var_name='', var_type='con'):
        """
        初始化对象
        :param X: 自变量
        :param y:  因变量
        :param var_name: 变量名称
        :param var_type: {'con', 'cate'} 变量类型
        """
        X = np.array(X)
        y = np.array(y)
        # 保证X，y有相同长度
        assert X.shape == y.shape

        self.X, self.y,self.var_name = X, y, var_name

        self.data = pd.DataFrame(zip(X, y),columns=['X','y'])
        self.var_type = var_type

    def describe(self) -> pd.DataFrame:
        """

        :return:
        """
        def _count_values(x):
            count_values = np.sum(pd.notna(x))
            f = count_values/len(x)
            return '%d (%.2f%%)' % (count_values, f*100)

        def _cal_unique(x):
            n = np.sum(~np.isnan(np.unique(x)))
            f = n / len(x)
            return '%d (%.2f%%)' % (n, f * 100)

        def _cal_nan(x):
            n= np.sum(np.isnan(x))
            f = n / len(x)
            return '%d (%.2f%%)' % (n, f * 100)

        if self.var_type == 'con':
            measures_1 = {
                'count': len,
                'values': _count_values,
                'unique': _cal_unique,
                'std': np.nanstd,
                'nan_values': _cal_nan,
                'median':np.nanmedian,
                'mean': np.nanmean

            }
            measures_2 ={
                'max': max,
                'min': min,
                'range': lambda x: max(x) - min(x),
                'var': np.nanvar,
                'kurt': lambda x: '%.2f' % (pd.Series(x).kurt()),
                'skew': lambda x: '%.2f' % (pd.Series(x).skew()),
                'sum': np.nansum
            }

            measures_values = {k:v(self.X) for k,v in measures_1.items()}
            df_1 = pd.DataFrame(measures_values,index=[0], dtype=object).T.reset_index()

            measures_values = {k:v(self.X) for k,v in measures_2.items()}
            df_2 = pd.DataFrame(measures_values,index=[0], dtype=object).T.reset_index()

            quan_marks = ['5%分位数','10分位数','25%分位数','50%分位数','75%分位数','90%分位数','95%分位数']
            quantiles = {k:v for k,v in zip(quan_marks,np.nanquantile(self.X, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))}
            df_3 = pd.DataFrame(quantiles,index=[0], dtype=object).T.reset_index()

            df = pd.concat([df_1, df_2, df_3], axis=1).fillna('')
            df.columns = [''] * 6
            return df
        else:
            from code_for_model_platform.cross_table import x_table
            return x_table(self.data, 'X', 'y')

    def plot_notebook(self, split_n=20):
        if self.var_type == 'con':
            return self.__plot_con_notebook(split_n)
        else:
            return self.__plot_cat_notebook()

    def __plot_cat_notebook(self):
        data = pd.value_counts(self.X)
        x = list(data.index.values)
        values = list(data.values / np.sum(data.values) * 100)

        bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        bar.load_javascript()

        bar.add_xaxis([i for i in x])

        bar.add_yaxis("1", [round(i, 2) for i in values])

        bar.set_global_opts(title_opts=opts.TitleOpts(title='self.var_name'),
                            xaxis_opts=opts.AxisOpts(name_location='end', name='%'),
                            yaxis_opts=opts.AxisOpts(name_location='end', name='Groups'),
                            )
        bar.reversal_axis()
        bar.set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="right"))

        return bar

    # todo
    def __plot_con_notebook(self, split_n)->pyecharts.charts.Bar:
        """
        调用echarts,连续变量
        :param split_n:
        :return:
        """
        data = self.data[~np.isnan(self.X)]
        data.sort_values('X',inplace=True)

        X = data['X'].values
        y = data['y'].values
        max_x = max(X)
        min_x = min(X)

        freq,edge = np.histogram(X, np.linspace(min_x, max_x, split_n+1))
        freq = np.round(freq/data.shape[0] *100,2)
        # print(freq)
        # print(edge)
        bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        bar.load_javascript()

        bar.add_xaxis([('（' + str(int(edge[i])) + ' , ' + str(int((edge[i + 1]))) + ']') if i else (
                    '[' + str(int(edge[i])) + ' , ' + str(int((edge[i + 1]))) + ']') for i in range(len(edge) - 1)])

        bar.add_yaxis(self.var_name, freq.tolist(), category_gap=1, )
        bar.set_global_opts(title_opts=opts.TitleOpts(title=self.var_name),
                           datazoom_opts=opts.DataZoomOpts(is_show=True),
                           xaxis_opts=opts.AxisOpts(name_location='end', name='Groups'),
                           yaxis_opts=opts.AxisOpts(name_location='end', name='%'),
                           )
        bar.set_series_opts(label_opts=opts.LabelOpts(is_show=True))
        return bar


if __name__ == '__main__':
    pass


