#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import stats
import seaborn as sns
# from scipy.stats import norm

plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.style.use('ggplot') # 使用ggplot风格

# 是否获得胜利
def is_win(rank):
    label = 0
    if rank == 1:
        label = 1
    return label

# 是否使用过车辆
def is_drive(distance):
    label = 0
    if distance != 0:
        label = 1
    return label


if __name__=='__main__':
    # 读入数据
    try:
        meta_data = pd.read_csv('agg_match_stats_0.csv',sep=',',header=0)
    except Exception as e:
        print('Error:',e)
    # 数据预览
    print('数据前五行')
    print(meta_data.head(5))
    print('数据总览')
    print(meta_data.describe())
    print('数据形状')
    print(meta_data.shape)
    print('数据列项')
    print(meta_data.columns)
    print('数据信息')
    print(meta_data.info)
    print('---------------------------')

    #选择分析的列数据
    used_data = meta_data[['match_id','game_size','match_mode','party_size','team_id','team_placement','player_kills',
    'player_dbno','player_assists','player_dmg','player_dist_ride','player_dist_walk','player_survive_time']]

    # 去除重复的比赛id数据，不过可能丢掉同一场比赛的多位玩家数据
    unique_match_data = used_data.drop_duplicates('match_id')
    unique_match_data.set_index(np.arange(unique_match_data.shape[0]),inplace=True) #重设index成排序
    unique_match_data.isnull().all() #检查各列是否有空值
    unique_match_data.to_csv('./unique_match_data.csv',index = False)

    # 添加是否获得胜利的列
    pro_unique_match_data = unique_match_data.copy()
    pro_unique_match_data['win_victory'] = unique_match_data['team_placement'].apply(is_win)
    # 添加是否使用过车辆的列
    pro_unique_match_data['has_drive_player'] = unique_match_data['player_dist_ride'].apply(is_drive)

    # 查看共有多少场比赛
    unique_match_counts = pd.value_counts(pro_unique_match_data['match_id']).count()
    print(r'共有比赛%d场(不含相同场次)' % unique_match_counts)
    print('')

    # 游戏模式统计
    match_mode_counts = pro_unique_match_data['match_mode'].value_counts()
    print('游戏模式占比')
    print(r'tpp(第三人称):%.2f%%' % (match_mode_counts * 100 / pro_unique_match_data.shape[0]))
    print('')

    # 玩家驾驶车辆行驶距离数据
    fig = plt.figure(figsize = (8,8))
    plt.subplot(2,1,1)
    x = pro_unique_match_data[pro_unique_match_data['player_dist_ride'] != 0]['player_dist_ride'] # 去除掉玩家使用车辆行驶距离为0的数据
    x[x>=13000].replace(13000) # 超过行驶距离13000的数据算作13000
    plt.hist(x,edgecolor='k',density=0,bins=150,facecolor='#1C7ECE',lw=1,alpha=.8) # 玩家驾驶车辆行驶距离分布
    plt.xlim(0,14000)
    plt.title('玩家驾驶车辆行驶距离分布',fontsize = 14)
    plt.xlabel('行驶距离',fontsize = 11)
    plt.ylabel('玩家人数',fontsize = 11)
    plt.text(10000, 550, '13000以上计入13000', style='italic', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 7}, fontsize=9)
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    no_drive_player = pro_unique_match_data['player_dist_ride'].value_counts()[0] / pro_unique_match_data.shape[0]
    print('游戏中没有驾驶过车辆玩家的占比为%.2f%%' % (no_drive_player*100))
    print('驾驶过车辆玩家的占比为%.2f%%' % ((1-no_drive_player) * 100))
    print('')
    # #sns.distplot(x,norm_hist =False,kde=True,hist_kws={'edgecolor':'k'}) #fit = norm,norm_hist无法使用
    plt.subplot(2,2,3)
    plt.title('玩家车辆使用状况')
    plt.pie([no_drive_player,1-no_drive_player],labels=['没有驾驶过车辆','驾驶过车辆'],autopct='%f%%',colors = ['#1C7ECE','#FF4D5B'],startangle = 120)
    plt.show()
    # 是否驾驶过车辆对获得胜利的影响
    plt.subplot(2, 2, 4)
    ct = pd.crosstab(index=pro_unique_match_data['win_victory'], columns=pro_unique_match_data['has_drive_player'])
    ct_df = pd.DataFrame(ct)
    plt.bar(ct_df.index,ct_df.loc[0]-ct_df.loc[1],label = '没获胜',width = 0.4,color = '#1C7ECE',alpha=.8)
    plt.bar(ct_df.index,ct_df.loc[1],bottom=(ct_df.loc[0]-ct_df.loc[1]),label='获胜',width = 0.4, color = '#FF4D5B',alpha=.8)
    plt.xlim(-1.1,2)
    plt.xticks(ct_df.index)
    plt.xlabel('是否驾驶过车辆')
    plt.legend()
    plt.title('是否驾驶过车辆对获得胜利的影响')
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.tight_layout()
    plt.savefig('./pic/drive_status.png',dpi = 300)
    plt.show()

    # 队伍规模频次统计
    fig = plt.figure(figsize=(7,7))
    x = pro_unique_match_data['party_size'].value_counts()
    plt.title('队伍规模场次统计')
    plt.xlabel(r'队伍规模(人)')
    plt.ylabel('出现频次')
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.bar(x.index,x,edgecolor='k',width = 0.3,facecolor ='#1C7ECE',alpha=.8)
    plt.xticks(x.index)
    plt.savefig('./pic/party_size.png',dpi = 300)
    plt.show()

    # 不同队伍规模的玩家生存时间
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('不同队伍规模下的表现',fontsize=15)
    plt.subplot(1,2,1)
    survive_time_group_by_size = pro_unique_match_data.groupby('party_size')['player_survive_time'].mean()
    survive_time_group_by_size_df = pd.DataFrame(survive_time_group_by_size).reset_index()
    plt.title('玩家生存时间')
    plt.xlabel(r'队伍规模(人)')
    plt.ylabel('生存时间')
    plt.bar(survive_time_group_by_size_df['party_size'],survive_time_group_by_size_df['player_survive_time'],edgecolor='k',width = 0.3,facecolor ='#1C7ECE',alpha=.8)
    plt.xticks(survive_time_group_by_size.index)
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.show()
    # 不同队伍规模的排名表现
    plt.subplot(1,2,2)
    party_size_group_by_placement =  pro_unique_match_data.groupby('party_size')['team_placement'].mean()
    print('不同队伍规模的平均排名')
    print(party_size_group_by_placement)
    party_size_group_by_placement_df = pd.DataFrame(party_size_group_by_placement).reset_index()
    plt.title('平均排名表现')
    plt.xlabel(r'队伍规模(人)')
    plt.ylabel('排名')
    plt.bar(party_size_group_by_placement_df['party_size'], party_size_group_by_placement_df['team_placement'], edgecolor='k', width=0.3, facecolor='#1C7ECE',alpha=.8)
    plt.xticks(survive_time_group_by_size.index)
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.savefig('./pic/party_size_vs_survive_time.png',dpi = 300)
    plt.show()

    # 取得单人模式比赛数据
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('各 模 式 下 击 杀 人 数 分 布',x=0.53,y=1)
    plt.subplot(2, 1, 1)
    single_player_match = unique_match_data.loc[unique_match_data['party_size'] == 1]
    # 单人模式下击杀统计
    x = single_player_match['player_kills']
    plt.title('单人模式击杀人数分布')
    plt.bar(x.value_counts().index.values, x.value_counts(), edgecolor='k', width=0.7, color='#1C7ECE', alpha=.8)
    # plt.xticks(x.value_counts().index.values)
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.xlim(0,20)
    plt.xlabel(r'击杀人数')
    plt.ylabel('击杀频次')
    plt.show()
    # 组队模式下击杀统计
    plt.subplot(2, 1, 2)
    team_player_match = pro_unique_match_data.loc[pro_unique_match_data['party_size'] != 1]
    x = team_player_match['player_kills']
    # sns.distplot(x, hist=True)
    plt.bar(x.value_counts().index.values, x.value_counts(), edgecolor='k', width=0.7, color='#1C7ECE', alpha=.8)
    plt.xlim(0, 20)
    plt.xlabel(r'击杀人数')
    plt.ylabel('击杀频次')
    plt.title('组队模式击杀人数分布')
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.tight_layout()
    plt.savefig('./pic/single_player_status.png',dpi = 300)
    plt.show()

    # 击杀人数和造成伤害与获得胜利的关系
    fig = plt.figure(figsize=(9, 5))
    pro_unique_match_data['获胜'] = pro_unique_match_data['win_victory']
    g = sns.stripplot(data=pro_unique_match_data[['获胜', 'player_dmg', 'player_kills']], x='player_kills',y='player_dmg', hue='获胜')
    g.set(title='击杀人数和伤害量与获得胜利的关系分布', xlabel='击杀人数', ylabel='造成的伤害')
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.savefig('./pic/kill_dmg_win.png',dpi = 300)
    plt.show()

    # 各变量与获胜的关系
    fig = plt.figure(figsize=(9, 7))
    plt.subplot(2, 1, 1)
    # 助攻与获得胜利的关系
    g = sns.barplot(data=pro_unique_match_data[['win_victory', 'player_assists']], x='player_assists', y='win_victory',color='#1C7ECE')
    g.set(title='助攻与获得胜利的关系', xlabel='助攻次数', ylabel='获胜概率密度')
    plt.grid(True, linestyle='--', linewidth=1, axis='y', alpha=0.4)
    # 击倒敌人与获得胜利的关系
    plt.subplot(2, 1, 2)
    g = sns.barplot(data=pro_unique_match_data[['win_victory', 'player_dbno']], x='player_dbno', y='win_victory',color='#1C7ECE')
    g.set(title='击倒敌人与获得胜利的关系', xlabel='击倒敌人次数', ylabel='获胜概率密度')
    plt.grid(True, linestyle='--', linewidth=1, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig('./pic/factors_vs_win.png', dpi=300)
    plt.show()








