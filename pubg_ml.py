import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 是否获得胜利
def is_win(rank):
    label = 0
    if rank == 1:
        label = 1
    return label


if __name__=='__main__':
    try:
        meta_data = pd.read_csv('agg_match_stats_0.csv',sep=',',header=0)
    except Exception as e:
        print('Error:',e)
        # 选择分析的列数据
    used_data = meta_data[ ['match_id', 'game_size',  'party_size', 'team_id', 'team_placement', 'player_kills',
         'player_dbno', 'player_assists', 'player_dmg', 'player_dist_ride', 'player_dist_walk', 'player_survive_time']]

    # 去除重复的比赛id数据，不过可能丢掉同一场比赛的多位玩家数据
    unique_match_data = used_data.drop_duplicates('match_id')
    unique_match_data.set_index(np.arange(unique_match_data.shape[0]), inplace=True)  # 重设index成排序

    pro_unique_match_data = unique_match_data.copy()
    # 是否获得胜利作为标签y
    pro_unique_match_data['label'] = unique_match_data['team_placement'].apply(is_win)

    # 选择数字型数据
    numeric_cols = ['game_size','player_kills','player_dbno','player_assists','player_dmg','player_dist_ride','player_dist_walk','player_survive_time']
    # 选择类别型数据
    category_cols = ['party_size',]
    # 选择标签列
    label_col = ['label']

    # 最后组成的数据集
    user_cols = numeric_cols + category_cols + label_col
    final_samples = pro_unique_match_data[user_cols]
    # 去掉空值
    final_samples.dropna(inplace=True)
    final_samples.to_csv('./ml_final_samples.csv',index=False)

    numeric_feat = final_samples[numeric_cols].values
    category_val = final_samples[category_cols].values[:, 0]  # 如果有多列，每次处理一列

    # 处理类别数据
    # label encoder
    # label_enc = preprocessing.LabelEncoder()
    # label_val = label_enc.fit_transform(category_val)
    label_val = category_val.reshape(-1, 1)
    # one-hot encoder
    onehot_enc = preprocessing.OneHotEncoder()
    category_feat = onehot_enc.fit_transform(label_val)
    category_feat = category_feat.toarray()

    # 生成最终特征和标签用于模型的训练
    X = np.hstack((numeric_feat, category_feat))
    y = final_samples[label_col].values

    # 数据集信息
    n_sample = y.shape[0]
    n_pos_sample = y[y == 1].shape[0]
    n_neg_sample = y[y == 0].shape[0]
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))
    print('特征维数：', X.shape[1])

    # 处理不平衡数据
    sm = SMOTE(random_state=42)
    X, y = sm.fit_sample(X, y.ravel())
    print('通过SMOTE方法平衡正负样本后')
    n_sample = y.shape[0]
    n_pos_sample = y[y == 1].shape[0]
    n_neg_sample = y[y == 0].shape[0]
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

    # 生成逻辑回归模型
    lr_model = LogisticRegression(C=1.0)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    # 查看得分
    accuracy = metrics.accuracy_score(y_pred, y_test)
    precision = metrics.precision_score(y_pred, y_test, pos_label=1)
    recall = metrics.recall_score(y_pred, y_test, pos_label=1)
    print('accuracy:',accuracy)
    print('precision:',precision)
    print('recall:',recall)