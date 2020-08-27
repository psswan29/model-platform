# 测试时间 2020年8月10日
# 测试环境 python 3.7
# pandas version 1.0.5
# 本文档为一个建模实例

from code_for_model_platform import *
from sklearn.model_selection import train_test_split

sns.set_style('whitegrid', {'font.sans-serif': ['simhei', 'Arial']})

# 设置路径 读入数据,
# 去掉各种键值，姓名、因变量
data_1 = pd.read_csv('lucheng_data.csv', encoding='gbk')
var_list = [i for i in  data_1.columns if i not in ['MOBILE_NUMBER', 'CERT_NO', 'NAME',
                                                    'INNET_DATE', 'UP_TIME','AREA_ID', 'Y']]

# 划分类别变量与连续变量
# todo
# var_type_dict = divide_cat_con_var(data_1, var_list)
var_type_dict = judge_leibie_lianxu(data_1[var_list])

# 缺失同质检验
# 类别变量分析 仅显示存在同质性的变量
# todo
discrete_1 = discrete_variable_table(data_1, var_type_dict['discrete'])
var_tongzhi_list_1 = discrete_variable_univar(discrete_1)

# 类别变量分布情况
for ii in discrete_1:
    vals = list(discrete_1[ii]['Proportion'])
    labels = list(discrete_1[ii].index)
    pie_drawing(vals, labels, ii)

# 剔除同质待分析的类别变量
var_discrete_analyse = [x for x in var_type_dict['discrete'] if x not in var_tongzhi_list_1]

# 连续变量同质分析
continua_1 = fenweishu_continuous_variable(data_1, var_type_dict['continua'])
var_tongzhi_list_2 = continua_variable_univar(continua_1)

# 可视化连续变量层级
visualize_continua_var_layers(var_type_dict, data_1)
# 剔除同质待分析的连续变量
var_continua_analyse = [x for x in var_type_dict['continua'] if x not in var_tongzhi_list_2]

# 画图展示相关系数热力图
draw_heat(data_1, var_continua_analyse)

# 检验连续变量相关系数
var_cor_75_dict = tst_continu_var(data_1, var_continua_analyse)
# 在这里由于暂时没有合适聚类算法，因此手动筛选变量
# todo
list_remove_1 = []
list_save = []
for j in var_cor_75_dict.keys():
    list_save.append(j)
    for j2 in var_cor_75_dict[j]:
        if j2[0] >= 0.75 and (j2[1] not in list_save):
            list_remove_1.append(j2[1])

# 剔除相关系数后待分析连续变量
var_continua_analyse_2 = [x for x in var_continua_analyse if x not in list_remove_1]

# 批量分析类别变量
for k in var_discrete_analyse:
    print(k + '\n', table_XXX(data_1, k, 'Y'), '\n')

# 批量分析连续变量
sns.set(style="darkgrid")
for k1 in var_continua_analyse_2:
    print(k1 + '\n')
    drawing(data_1, k1, 'Y')
    print('\n')

# 类别变量处理
data_1['GENDER_1'] = (data_1['GENDER'] == 1).astype(int)
data_1['PAY_MODE_1'] = (data_1['GENDER'].apply(lambda x: x in [2, 5])).astype(int)
data_1['SERVICE_TYPE_1'] = (data_1['SERVICE_TYPE'] == '200101AA').astype(int)
data_1['GROUP_FLAG_1'] = (data_1['GROUP_FLAG'] == 1).astype(int)
data_1['USER_STATUS_1'] = (data_1['USER_STATUS'] == 11).astype(int)
data_1['FACTORY_DESC_1'] = (data_1['FACTORY_DESC'] == '苹果').astype(int)
data_1['DEV_CHANGE_NUM_Y1_1'] = (data_1['DEV_CHANGE_NUM_Y1'].apply(lambda x: x in [4, 5, 6])).astype(int)
data_1['REAL_HOME_FLAG_M1_1'] = (data_1['REAL_HOME_FLAG_M1'] == 1).astype(int)
data_1['LIKE_HOME_FLAG_M1_1'] = (data_1['LIKE_HOME_FLAG_M1'] == 1).astype(int)
data_1['REAL_WORK_FLAG_M1_1'] = (data_1['REAL_WORK_FLAG_M1'] == 1).astype(int)
data_1['LIKE_WORK_FLAG_M1_1'] = (data_1['LIKE_WORK_FLAG_M1'] == 1).astype(int)

var_discrete_for_model = ['GENDER_1', 'PAY_MODE_1', 'SERVICE_TYPE_1', 'GROUP_FLAG_1',
                          'USER_STATUS_1', 'FACTORY_DESC_1', 'DEV_CHANGE_NUM_Y1_1',
                          'REAL_HOME_FLAG_M1_1', 'LIKE_HOME_FLAG_M1_1', 'REAL_WORK_FLAG_M1_1'
                         ,'LIKE_WORK_FLAG_M1_1']

# todo
# 连续变量自动处理
var_continua_for_model, var_continua_process = auto_deal_continua(var_continua_analyse_2, data_1)

# 汇总所有解释变量
var_for_model_all = var_discrete_for_model + var_continua_for_model
var_for_model_all_y =  var_for_model_all+ ['Y']
X_train, X_test, y_train, y_test = train_test_split(data_1.drop('Y', 1), data_1['Y'], test_size=0.2,
                                                    random_state=123456)
train, test = train_test_split(data_1, test_size=0.2, random_state=123456)

# 构造逐步回归筛选变量并建模
model_final = stepwise_selection(train[var_for_model_all_y],sle=0.15, sls=0.15)
# 构造反向淘汰回归筛选变量并建模
model_final2 = backward_selection(train[var_for_model_all_y])


# 显示模型结果
model_final.summary()
model_final2.summary()

# AUC曲线和C值、GINI
y_pred = model_final.predict()
# 测试集
y_test_pred = model_final.predict(test[var_for_model_all])

AUC_GINI_KS.roc_auc_gini(train['Y'], y_pred)
AUC_GINI_KS.roc_auc_gini(test['Y'], y_test_pred)


# ks值
print('KS:', AUC_GINI_KS.get_ks(y_pred, y_train))
print('KS:', AUC_GINI_KS.get_ks(y_test_pred, y_test))

# f-1检验
f1_result = f1_test(train['Y'], y_pred)
f1_result2 = f1_test_m(train['Y'], y_pred)

# 模型十等分
print("模型的十等分：\n", model_10_splitm(model_final, train))

# 预测值转化为评分卡评分
print(transfer_score(y_pred))
# 模型系数转换为评分卡
print(build_score_card(model_final.params, step_score=50))