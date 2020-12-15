import pandas as pd
import numpy as np
from code_for_model_platform.homogeneity_test import discrete_variable_table, discrete_variable_univar,continua_variable_univar_m
from code_for_model_platform.corr_test import  tst_continu_var_1
from code_for_model_platform.varcluster import VarClusHi
from code_for_model_platform.ginni_split import auto_ginni_split
from code_for_model_platform.auto_bin_using_chi import auto_deal_continua
from sklearn.model_selection import train_test_split
from code_for_model_platform.modeling import stepwise_selection,backward_selection


class modeling(object):

    def __init__(self, df: pd.DataFrame,
                 target_var:str,
                 var_del=[],
                 var_discrete=[],
                 var_continual=[],
                 method ='LG',
                 random_state=None
                 ):
        if (df.isna().values).all():
            raise ValueError("All values in `df` must be non-nan .")

        self.df = df
        self.y = target_var
        self.var_discrete = [col for col in var_discrete if col not in var_del]
        self.var_continual = [col for col in var_continual if col not in var_del]
        self.method = method
        self.random_state = random_state


    def fit(self, verbose=True, LG_type='step',sle=0.15, sls=0.15, tst_split=False,**kwargs):
        global seed
        seed = np.random.RandomState(self.random_state)
        print()
        print('正在进行---类别变量同质性检验。。。。。。')
        discrete_1 = discrete_variable_table(self.df, self.var_discrete)
        var_tongzhi_list_1 = discrete_variable_univar(discrete_1)
        # 剔除同质待分析的类别变量
        var_discrete_analyse = [x for x in self.var_discrete if x not in var_tongzhi_list_1[1]]

        print()
        print('正在进行---连续变量同质性检验。。。。。。')
        var_tongzhi_list_2 = continua_variable_univar_m(self.df, self.var_continual)
        var_continua_analyse = [x for x in self.var_continual if x not in var_tongzhi_list_2[1]]

        print()
        print('正在进行---第一次连续变量相关性检验。。。。。。')
        corr_dict, corr_, independent_var,log = tst_continu_var_1(self.df, var_continua_analyse, corr_rate=0.75)
        keep_vars1 = VarClusHi.reduce_dimension(self.df, corr_, verbose=verbose)
        list_remove_1 = ('第一次连续变量相关性检验', [e for e in {i for s in corr_.values() for i in s} if e not in keep_vars1])
        # 剔除相关系数后待分析连续变量
        var_continua_analyse_2 = [x for x in var_continua_analyse if x not in list_remove_1[1]]

        print()
        print('正在进行---类别变量自动分箱。。。。。。')

        # todo gini分箱存在bug，需修改
        # 此时data_1
        data_1, new_col = auto_ginni_split(self.df,
                                           conti_var=[],
                                           cate_var=var_discrete_analyse,
                                           y_name=self.y,
                                           gaps=0.05,
                                           verbose=verbose)

        var_discrete_for_model = new_col

        print()
        print('正在进行---连续变量自动分箱。。。。。。')
        # 连续变量自动处理
        var_continua_for_model, var_continua_process = auto_deal_continua(var_continua_analyse_2, data_1,y_name=self.y)

        # 汇总所有解释变量
        var_for_model_all = var_discrete_for_model + var_continua_for_model

        print()
        print('正在进行---入模变量相关性检验。。。。。。')
        corr_dict, corr_, independent_var, log = tst_continu_var_1(data_1, var_for_model_all)

        keep_vars2 = VarClusHi.reduce_dimension(data_1, corr_, verbose=True)
        var_del2 = ('入模变量相关性检验', [e for e in {i for s in corr_.values() for i in s} if e not in keep_vars2])

        var_for_model_all_y = keep_vars2 + independent_var + [self.y]
        X_train, X_test, y_train, y_test = train_test_split(data_1.drop(self.y, 1), data_1[self.y], test_size=0.2,
                                                            random_state=1234590)

        # 是否划分训练集、测试集
        train = data_1
        if tst_split:
            train, test = train_test_split(data_1, test_size=0.2, random_state=1234590)
            y_test_pred = self.model_final.predict(test[var_for_model_all])

        print()
        print('正在进行---建模过程。。。。。。')
        # 构造逐步回归筛选变量并建模
        if LG_type in ('step', 'STEP'):
            self.model_final = stepwise_selection(train[var_for_model_all_y],y_n=self.y, sle=sle, sls=sls, verbose=verbose)
        else:
            self.model_final = backward_selection(train[var_for_model_all_y],y_name=self.y, sle=sle, sls=sls, verbose=verbose)

        y_pred = self.model_final.predict()


        from code_for_model_platform.AUC_GINI_KS import roc_auc_gini,get_ks

        print('训练集auc：', roc_auc_gini(train[self.y], y_pred))
        print('训练集KS:', get_ks(y_pred, train[self.y]))

        if tst_split:
            print('测试集auc：', roc_auc_gini(test[self.y], y_test_pred))
            print('测试集KS:', get_ks(y_test_pred, y_test))

        from code_for_model_platform.F1test import  f1_test_m
        # f-1检验
        f1_result2 = f1_test_m(train[self.y], y_pred, verbose=verbose)

        from code_for_model_platform.ten_split import model_10_splitm
        # 模型十等分
        print("模型的十等分：\n", model_10_splitm(self.model_final, train,target_n=self.y))

