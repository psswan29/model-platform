import pandas as pd
import numpy as np
from code_for_model_platform.homogeneity_test import discrete_variable_table, discrete_variable_univar,continua_variable_univar_m
from code_for_model_platform.corr_test import  tst_continu_var_1
from code_for_model_platform.varcluster import VarClusHi
from code_for_model_platform.ginni_split import auto_ginni_split
from code_for_model_platform.auto_bin_using_chi import auto_deal_continua
from sklearn.model_selection import train_test_split
from code_for_model_platform.modeling import stepwise_selection,backward_selection
from code_for_model_platform.ten_split import model_10_splitm
from code_for_model_platform.AUC_GINI_KS import roc_auc_gini, get_ks
from code_for_model_platform.var_transfer import var_change

class modeling(object):

    def __init__(self, df: pd.DataFrame,
                 target_var:str,
                 var_del:list,
                 var_discrete:list,
                 var_continual:list,
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


    def fit(self, verbose=True, LG_type='step',sle=0.15, sls=0.15, tst_split=False, **kwargs):

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
        data_1, new_col,cate_process_dict = auto_ginni_split(self.df,
                                                             conti_var=[],
                                                             cate_var=var_discrete_analyse,
                                                             y_name=self.y,
                                                             gaps=0.05,
                                                             verbose=verbose)
        self.cate_process_dict = cate_process_dict
        var_discrete_for_model = new_col

        print()
        print('正在进行---连续变量自动分箱。。。。。。')
        # 连续变量自动处理
        var_continua_for_model, var_continua_process = auto_deal_continua(var_continua_analyse_2, data_1,y_name=self.y,verbose=verbose)
        self.var_continua_process = var_continua_process

        print()
        print('正在进行---连续变量自动拟合。。。。。。')
        var_fittings = [var_change(data_1, col[:-2], self.y, n=20) for col in var_continua_for_model]

        self.var_fitting_process = {}

        if any(var_fittings):
            for i in var_fittings:
                if not i: continue
                var, feature, series = i
                print('%s is added to training set' % (var+feature))
                data_1[var+feature] = series
                var_continua_for_model.append(var+feature)
                self.var_fitting_process[var]=feature

        print()
        print('正在进行---入模变量相关性检验。。。。。。')
        # 汇总所有解释变量
        var_for_model_all = var_discrete_for_model + var_continua_for_model
        corr_dict, corr_, independent_var, log = tst_continu_var_1(data_1, var_for_model_all)

        keep_vars2 = VarClusHi.reduce_dimension(data_1, corr_, verbose=verbose)
        var_del2 = ('入模变量相关性检验', [e for e in {i for s in corr_.values() for i in s} if e not in keep_vars2])

        var_for_model_all_ = keep_vars2 + independent_var
        self.var_for_model_all_ = var_for_model_all_
        var_for_model_all_y = var_for_model_all_ + [self.y]
        print()
        print('入模变量为',var_for_model_all_y)
        print()
        # 是否划分训练集、测试集
        train = data_1
        if tst_split:
            train, test = train_test_split(data_1, test_size=0.2, random_state=1234590)

        print()
        print('正在进行---建模过程。。。。。。')

        # 若给定褚时列表则依据给定列表设置入模变量
        # todo
        initial_list = kwargs.get('initial_list',[])
        max_var_num = kwargs.get('max_var_num', 20)
        # 构造逐步回归筛选变量并建模
        if LG_type in ('step', 'STEP'):
            self.model_final = stepwise_selection(train[var_for_model_all_y],initial_list=initial_list,y_n=self.y, sle=sle, sls=sls, verbose=verbose,max_var_num=max_var_num)
        else:
            self.model_final = backward_selection(train[var_for_model_all_y],y_name=self.y, sle=sle, sls=sls, verbose=verbose)

        print()
        print('正在汇总建模过程变量删除情况')
        self.__print_model_process( var_tongzhi_list_1, var_tongzhi_list_2, list_remove_1, var_del2)

        y_pred = self.model_final.predict(train[var_for_model_all_])

        print()
        print('正在打印模型评价指标')

        self.auc = roc_auc_gini(train[self.y], y_pred)
        self.ks = get_ks(y_pred, train[self.y])
        print('训练集auc：', self.auc)
        print('训练集KS:', self.ks)


        from code_for_model_platform.F1test import  f1_test_m
        # f-1检验
        self.f1_result = f1_test_m(train[self.y], y_pred, verbose=verbose)

        # 模型十等分
        self.ten_split = model_10_splitm(self.model_final, train,target_n=self.y)
        print("模型的十等分：\n", self.ten_split)

        if tst_split:
            y_test_pred = self.model_final.predict(test[var_for_model_all])
            print('测试集auc：', roc_auc_gini(test[self.y], y_test_pred))
            print('测试集KS:', get_ks(y_test_pred, test[self.y]))
            print("模型的十等分：\n", model_10_splitm(self.model_final, test, target_n=self.y))

    def eval(self, eval_X:pd.DataFrame):
        df_1 = {}
        df_1["_sq"] = lambda X: X ** 2
        df_1["_sqrt"] = lambda X: np.sqrt(np.where(X < 0, 0, X))  # sqrt(max(x,0))
        df_1["_cu"] = lambda X: X ** 3
        df_1["_curt"] = lambda X: np.cbrt(X)
        df_1["_log"] = lambda X: np.log(np.where(X < 1e-3, 1e-3, X))

        for col in  eval_X.columns:
            if col in self.cate_process_dict.keys():
                eval_X[col + '_1'] = eval_X[col].isin(self.cate_process_dict[col]).astype(int)
                continue

            if col in self.var_continua_process.keys():
                edges = [float(i.strip()) for i in self.var_continua_process[col].replace('<','').replace('>','').replace('=','').split('and')]
                if len(edges)==2:
                    eval_X[col + '_1'] = ((eval_X[col]>edges[0]) & (eval_X[col]<=edges[1])).astype(int)
                else:
                    eval_X[col + '_1'] = (eval_X[col]<=edges[0]).astype(int)

            if col in self.var_fitting_process.keys():
                feature =  self.var_fitting_process[col]
                f = df_1[feature]
                eval_X[col+feature] = f(eval_X[col])


        self.eval_pred = self.model_final.predict(eval_X[self.var_for_model_all_])
        self.eval_auc = roc_auc_gini(eval_X[self.y], self.eval_pred)
        self.eval_ks = get_ks(self.eval_pred, eval_X[self.y])
        self.eval_ten_split = model_10_splitm(self.model_final, eval_X, target_n=self.y)

        print('测试集auc：', self.eval_auc)
        print('测试集KS:', self.eval_ks)
        print("模型的十等分：\n", self.eval_ten_split)


    # 打印变量处理过程
    def __print_model_process(self, *args):
        for i,e in enumerate(args):
            print('-'*200)
            print(f'建模流程 第{i+1}步 {e[0]} 删除的变量有: ')
            print('-' * 200)
            for v in e[1]:
                print(v,end='\t')
            print(end='\n')
            print('-' * 200)


if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    from code_for_model_platform_cls.auto_modeling import modeling

    df = pd.read_csv('../data/sjf_train_data.csv')

    str1 = """AGE
    STOP_MONTH
    CERT_NUMS
    INTER_LONG_FEE_FIRST
    INTER_LONG_FEE_SECOND
    INTER_LONG_FEE_THIRD
    ROAM_NUM_FIRST
    OUT_DURA_M1
    OUT_NUM_M1
    IN_DURA_M1
    IN_NUM_M1
    OUT_COUNROAM_NUM_M1
    OUT_INTERROAM_NUM_M1
    IN_COUNROAM_NUM_M1
    IN_INTERROAM_NUM_M1
    OUT_COUNLONG_NUM_M1
    OUT_INTERLONG_NUM_M1
    NOROAM_LOCAL_NUM_M1
    ROAM_LOCAL_NUM_M1
    NOROAM_COUNLONG_NUM_M1
    ROAM_COUNLONG_NUM_M1
    FLUX_NUM_M1
    FREE_FLUX_NUM_M1
    BILL_FLUX_NUM_M1
    LOCAL_FLUX_NUM_M1
    COUN_FLUX_NUM_M1
    INTER_FLUX_NUM_M1
    TOTAL_FLUX_M1
    FREE_FLUX_M1
    BILL_FLUX_M1
    TOTAL_FLUX_DURA_M1
    FREE_FLUX_DURA_M1
    BILL_FLUX_DURA_M1
    FLUX_FEE_M1
    PROD_IN_LOCAL_FLUX_M1
    PROD_IN_COUN_FLUX_M1
    PROD_IN_INTER_FLUX_M1
    DEV_CHANGE_NUM_Y1
    CONTACT_CNT_M1
    NIGHT_IN_CNT_M1
    NIGHT_OUT_CNT_M1
    NIGHT_CNT_M1
    SHIPIN_APP_NUM_M1
    SP_VISIT_CNT_M1
    SP_ACTIVE_MAX_DAYS_M1
    YINPIN_APP_NUM_M1
    YP_VISIT_CNT_M1
    YP_ACTIVE_MAX_DAYS_M1
    WEIXIN_APP_NUM_M1
    WX_VISIT_CNT_M1
    WX_ACTIVE_MAX_DAYS_M1
    QQ_APP_NUM_M1
    QQ_VISIT_CNT_M1
    QQ_ACTIVE_MAX_DAYS_M1
    MILIAO_APP_NUM_M1
    ML_VISIT_CNT_M1
    ML_ACTIVE_MAX_DAYS_M1
    GOUWU_APP_NUM_M1
    GW_VISIT_CNT_M1
    GW_ACTIVE_MAX_DAYS_M1
    ZHIFUBAO_APP_NUM_M1
    ZFB_VISIT_CNT_M1
    ZFB_ACTIVE_MAX_DAYS_M1
    SJYH_APP_NUM_M1
    SJYH_VISIT_CNT_M1
    SJYH_ACTIVE_MAX_DAYS_M1
    WAIMAI_APP_NUM_M1
    WM_VISIT_CNT_M1
    WM_ACTIVE_MAX_DAYS_M1
    TUANGOU_APP_NUM_M1
    TG_VISIT_CNT_M1
    TG_ACTIVE_MAX_DAYS_M1
    MUYING_APP_NUM_M1
    MY_VISIT_CNT_M1
    MY_ACTIVE_MAX_DAYS_M1
    MAIL_APP_NUM_M1
    MAIL_VISIT_CNT_M1
    MAIL_ACTIVE_MAX_DAYS_M1
    ZHENGQUAN_APP_NUM_M1
    ZQ_VISIT_CNT_M1
    ZQ_ACTIVE_MAX_DAYS_M1
    BAOXIAN_APP_NUM_M1
    BX_VISIT_CNT_M1
    BX_ACTIVE_MAX_DAYS_M1
    LICAI_APP_NUM_M1
    LC_VISIT_CNT_M1
    LC_ACTIVE_MAX_DAYS_M1
    XINYONGKA_APP_NUM_M1
    XYK_VISIT_CNT_M1
    XYK_ACTIVE_MAX_DAYS_M1
    CAIPIAO_APP_NUM_M1
    CP_VISIT_CNT_M1
    CP_ACTIVE_MAX_DAYS_M1
    DAIKUAN_APP_NUM_M1
    DK_VISIT_CNT_M1
    DK_ACTIVE_MAX_DAYS_M1
    DAIKUAN1_APP_NUM_M1
    DK1_VISIT_CNT_M1
    DK1_ACTIVE_MAX_DAYS_M1
    DAIKUAN2_APP_NUM_M1
    DK2_VISIT_CNT_M1
    DK2_ACTIVE_MAX_DAYS_M1
    DAIKUAN3_APP_NUM_M1
    DK3_VISIT_CNT_M1
    DK3_ACTIVE_MAX_DAYS_M1
    REAL_HOME_FLAG_M1
    LIKE_HOME_FLAG_M1
    REAL_WORK_FLAG_M1
    LIKE_WORK_FLAG_M1
    ST_MAXDURA_PERDAY_M1
    ST_NUM_M1
    ST_NUM_PERDAY_M1
    HOME_ST_NUM_M1
    HOME_ST_NUM_PERDAY_M1
    WORK_ST_NUM_M1
    WORK_ST_NUM_PERDAY_M1
    MIDNIGHT_NUM_M1
    MIDNIGHT_IN_NUM_M1
    MIDNIGHT_OUT_NUM_M1
    MIDNIGHT_CNT_M1
    MIDNIGHT_IN_CNT_M1
    MIDNIGHT_OUT_CNT_M1
    MIDNIGHT_DURA_M1
    WORK_NIGHT_NUM_M1
    WORK_NIGHT_IN_NUM_M1
    WORK_NIGHT_OUT_NUM_M1
    WORK_NIGHT_CNT_M1
    WORK_NIGHT_IN_CNT_M1
    WORK_NIGHT_OUT_CNT_M1
    WORK_NIGHT_DURA_M1
    WEEKEND_NIGHT_NUM_M1
    WEEKEND_NIGHT_IN_NUM_M1
    WEEKEND_NIGHT_OUT_NUM_M1
    WEEKEND_NIGHT_CNT_M1
    WEEKEND_NIGHT_IN_CNT_M1
    WEEKEND_NIGHT_OUT_CNT_M1
    WEEKEND_NIGHT_DURA_M1
    WORK_DAY_NUM_M1
    WORK_DAY_IN_NUM_M1
    WORK_DAY_OUT_NUM_M1
    WORK_DAY_CNT_M1
    WORK_DAY_IN_CNT_M1
    WORK_DAY_OUT_CNT_M1
    WORK_DAY_DURA_M1
    WEEKEND_DAY_NUM_M1
    WEEKEND_DAY_IN_NUM_M1
    WEEKEND_DAY_OUT_NUM_M1
    WEEKEND_DAY_CNT_M1
    WEEKEND_DAY_IN_CNT_M1
    WEEKEND_DAY_OUT_CNT_M1
    WEEKEND_DAY_DURA_M1
    WORK_NIGHT_DURA_AVG_M1
    WEEKEND_NIGHT_DURA_AVG_M1
    WORK_DAY_DURA_AVG_M1
    WEEKEND_DAY_DURA_AVG_M1
    MIDNIGHT_DURA_AVG_M1
    SHORT_LOCAL_CALLING_NUMS
    SHORT_LOCAL_CALLED_NUMS
    SHORT_ROAM_COUN_CALLING_NUMS
    SHORT_ROAM_INTER_CALLING_NUMS
    SHORT_ROAM_COUN_CALLED_NUMS
    SHORT_ROAM_INTER_CALLED_NUMS
    SHORT_TOLL_COUN_CALLING_NUMS
    SHORT_TOLL_INTER_CALLING_NUMS
    MID_LOCAL_CALLING_NUMS
    MID_LOCAL_CALLED_NUMS
    MID_ROAM_COUN_CALLING_NUMS
    MID_ROAM_INTER_CALLING_NUMS
    MID_ROAM_COUN_CALLED_NUMS
    MID_ROAM_INTER_CALLED_NUMS
    MID_TOLL_COUN_CALLING_NUMS
    MID_TOLL_INTER_CALLING_NUMS
    LONG_LOCAL_CALLING_NUMS
    LONG_LOCAL_CALLED_NUMS
    LONG_ROAM_COUN_CALLING_NUMS
    LONG_ROAM_INTER_CALLING_NUMS
    LONG_ROAM_COUN_CALLED_NUMS
    LONG_ROAM_INTER_CALLED_NUMS
    LONG_TOLL_COUN_CALLING_NUMS
    LONG_TOLL_INTER_CALLING_NUMS
    SHORT_LOCAL_CALLING_COUNTS
    SHORT_LOCAL_CALLED_COUNTS
    SHORT_ROAM_COUN_CALLING_COUNTS
    SHORT_ROAM_INTER_CALLING_COUNTS
    SHORT_ROAM_COUN_CALLED_COUNTS
    SHORT_ROAM_INTER_CALLED_COUNTS
    SHORT_TOLL_COUN_CALLING_COUNTS
    SHORT_TOLL_INTER_CALLING_COUNTS
    MID_LOCAL_CALLING_COUNTS
    MID_LOCAL_CALLED_COUNTS
    MID_ROAM_COUN_CALLING_COUNTS
    MID_ROAM_INTER_CALLING_COUNTS
    MID_ROAM_COUN_CALLED_COUNTS
    MID_ROAM_INTER_CALLED_COUNTS
    MID_TOLL_COUN_CALLING_COUNTS
    MID_TOLL_INTER_CALLING_COUNTS
    LONG_LOCAL_CALLING_COUNTS
    LONG_LOCAL_CALLED_COUNTS
    LONG_ROAM_COUN_CALLING_COUNTS
    LONG_ROAM_INTER_CALLING_COUNTS
    LONG_ROAM_COUN_CALLED_COUNTS
    LONG_ROAM_INTER_CALLED_COUNTS
    LONG_TOLL_COUN_CALLING_COUNTS
    LONG_TOLL_INTER_CALLING_COUNTS
    CALLING_BANK_COUNT
    CALLING_BANK_DURA
    CALLED_BANK_COUNT
    CALLED_BANK_DURA
    CALLING_BANK_NUMS
    CALLED_BANK_NUMS
    LOCAL_CALLING_COUNTS
    LOCAL_CALLED_COUNTS
    ROAM_COUN_CALLING_COUNTS
    ROAM_INTER_CALLING_COUNTS
    ROAM_COUN_CALLED_COUNTS
    ROAM_INTER_CALLED_COUNTS
    TOLL_COUN_CALLING_COUNTS
    TOLL_INTER_CALLING_COUNTS
    TOTAL_SMS_NUM
    PTP_SMS_NUM
    SP_SMS_NUM
    OUT_SMS_NUM
    IN_SMS_NUM
    FREE_SMS_NUM
    BILL_SMS_NUM
    PTP_SMS_FEE
    SP_SMS_FEE
    TOTAL_SMS_FEE
    PTP_SMS_RATE
    SP_SMS_RATE
    IN_SMS_RATE
    FREE_SMS_RATE
    PTP_OUT_FREE_RATE
    PTP_OUT_CMCC_RATE
    PTP_SMS_FEE_RATE
    PTP_SMS_IN_RATE
    SMS_NUM_AVG_M3
    SMS_NUM_RATE_M1
    innet_month
    usage_month
    total_fee_m1
    total_fee_m3
    out_call_fee_m1
    out_call_fee_m3
    out_flux_fee_m1
    out_flux_fee_m3
    INCR_FEE_m1
    """
    str2 = """CERT_TYPE
    GENDER
    PAY_MODE
    SERVICE_TYPE
    GROUP_FLAG
    INNET_FLAG
    USER_GROUP_FLAG
    MINOR_ENTERPRISES_FLAG
    CUST_SIZE
    factory_desc_new
    mobile_prov
    """
    con = [col.strip() for col in str1.split('\n') if col]
    dis = [col.strip() for col in str2.split('\n') if col]

    model14 = modeling(df, 'Y',
                       var_del=['Y', '_TEMA001'],
                       var_discrete=dis,
                       var_continual=con)
    model14.fit(verbose=True, sle=0.05, sls=0.05, tst_split=False)
    s = model14.model_final.summary()









