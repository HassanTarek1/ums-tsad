import numpy as np
import pandas as pd
import sys
import joblib

from model_selection.rank_aggregation import trimmed_borda, trimmed_kemeny, trimmed_partial_borda, borda, kemeny, partial_borda, influence, averagedistance
from metrics.ranking_metrics import rank_by_praucs, rank_by_centrality, rank_by_synthetic_anomlies, rank_by_forecasting_metrics, rank_by_max_F1,rank_by_metrics
from utils.model_selection_utils import rank_models



result_path = "E:/zfsauton/project/public/Mononito/results/anomaly_archive/ranking_obj_074_UCR_Anomaly_DISTORTEDqtdbSel1005V.data"
rankingObj = joblib.load(result_path)
models_performance_matrix =rankingObj.models_performance_matrix
print(f'type(models_performance_matrix) is {type(models_performance_matrix)}')
print(f'models_performance_matrix is {models_performance_matrix}')
ranks_by_metrics, rank_prauc, rank_f1, rank_vus = rank_models(models_performance_matrix)
ranks = np.concatenate([ranks_by_metrics[:6, :], ranks_by_metrics[6::2, :]], axis=0).astype(int)
model_names = models_performance_matrix.index.values
print(f'model_names is {model_names}')
print(f'ranks is {ranks}')
df_rank = pd.DataFrame(ranks,columns=model_names,index=model_names)
print(f'df_rank is {df_rank}')
f1 = models_performance_matrix.iloc[:, 1].to_numpy()
print(f'f1[rank_f1.astype(int)] is {f1[rank_f1.astype(int)]}')

print(f'f1[borda(ranks)[1].astype(int)] is {f1[borda(ranks)[1].astype(int)]}')

print(f'-----------------------------')
print(f1[trimmed_borda(ranks = ranks, aggregation_type='borda', metric='influence')[1].astype(int)])
print(f1[partial_borda(ranks, top_k=5)[1].astype(int)])
print(f1[trimmed_partial_borda(ranks, top_k=5, metric='influence', aggregation_type='borda')[1].astype(int)])

print(f'-----------------------------')
top_k=5
reliability = influence(ranks, aggregation_type='borda', top_k=top_k)
# trimmed_ranks = ranks[reliability > 0, :]
trimmed_ranks = ranks[np.argsort(-1*reliability)[:1], :]
print(f1[borda(trimmed_ranks)[1].astype(int)])

top_k=5
for i, r in enumerate(ranks):
    print(i, f1[r][:5], r[:5], reliability[i])



from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=2, linkage='single').fit_predict(reliability.reshape((-1, 1)))

cluster_ids, counts = np.unique(clustering, return_counts=True)
largest_cluster_idx = cluster_ids[np.argmax(counts)] # Largest cluster
print(f'Clustering: ', clustering)

most_reliable_cluster_idx = np.argmax([
    np.sum(reliability[np.where(clustering == 0)[0]]),
    np.sum(reliability[np.where(clustering == 1)[0]])])
    # np.sum(reliability[np.where(clustering == 2)[0]])])
print('Most reliable cluster idx:', most_reliable_cluster_idx, f'Largest: {largest_cluster_idx}')

# trimmed_ranks = ranks[np.where(clustering == largest_cluster_idx)[0], :]
# trimmed_ranks = ranks[np.where(clustering == most_reliable_cluster_idx)[0], :]
trimmed_ranks = ranks[reliability > 0, :]
print(f'trimmed_ranks.shape is {trimmed_ranks.shape}')

print(f1[trimmed_partial_borda(ranks, top_k=5, aggregation_type='partial_borda', metric='influence')[1].astype(int)])

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(30, 5))

model_name = 'RNN_3'
start = np.argmax(rankingObj.predictions[model_name]['anomaly_labels'].flatten()) - 500
stop = np.argmax(rankingObj.predictions[model_name]['anomaly_labels'].flatten()) + 500

axes.plot(rankingObj.predictions[model_name]['Y'].flatten()[start:stop], label='Y')
axes.plot(rankingObj.predictions[model_name]['Y_hat'].flatten()[start:stop], label='Y_hat')

entity_scores = rankingObj.predictions[model_name]['entity_scores'].flatten()[start:stop]
entity_scores = (entity_scores - entity_scores.min())/(entity_scores.max() - entity_scores.min())
axes.plot(entity_scores, linestyle='--', c='magenta', label='Anomaly scores')
axes.legend()
plt.show()
# print('Best F-1\n',models_performance_matrix.loc[:, 'Best F-1'].sort_values())
print('LIKELIHOOD\n',models_performance_matrix.loc[:, 'LIKELIHOOD'].sort_values())

# anomaly_type = 'flip_rep_1_2'
# print('flip_rep_1_2\n',models_performance_matrix.loc[:, f"SYNTHETIC_F1_{anomaly_type.split('_')[0]}_type_{anomaly_type}"].sort_values())
anomaly_type = 'flip'
# print('flip\n',set([ '_'.join(i.split('_')[2:]) for i in synthetic_predictions['RNN_4'].keys() if anomaly_type in i]))

fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(30, 5))

axes.plot(rankingObj.synthetic_predictions[model_name][f'Ta_type_{anomaly_type}'].flatten(), label='T_a')
entity_scores = rankingObj.synthetic_predictions[model_name][f'entityscores_type_{anomaly_type}'].flatten()
entity_scores = (entity_scores - entity_scores.min())/(entity_scores.max() - entity_scores.min())
axes.plot(entity_scores, label='Anomaly Scores', linestyle='--', c='magenta')
axes.plot(rankingObj.synthetic_predictions[model_name][f'anomalylabels_type_{anomaly_type}'].flatten(), label='Anomaly Labels')
axes.legend()
plt.show()

print(rankingObj.synthetic_predictions['RNN_4'].keys())

anomaly_type = 'wander'
# print(models_performance_matrix.loc[:, [i for i in models_performance_matrix.columns if ((anomaly_type in i) and ('F1' in i))]].sort_values(by='SYNTHETIC_F1_wander_type_wander_rep_1_14'))

# ranks_by_metrics, rank_prauc, rank_f1 = rankingObj.rank_models()
# Ranking data based on metrics
n_metrics, n_models = ranks_by_metrics.shape
print(f'Number of metrics: {n_metrics} | Number of models: {n_models}')

from sklearn.cluster import AgglomerativeClustering
from model_selection.rank_aggregation import borda, influence, proximity, pagerank, kemeny, trimmed_borda, trimmed_kemeny
from sklearn.metrics import ndcg_score

influence_of_ranks = influence(ranks_by_metrics, aggregation_type='kemeny')
clustering = AgglomerativeClustering(n_clusters=2, linkage='single').fit_predict(influence_of_ranks.reshape((-1, 1)))

cluster_ids, counts = np.unique(clustering, return_counts=True)
max_cluster = cluster_ids[np.argmax(counts)]

_, trimmed_kemeny_rank = kemeny(ranks_by_metrics[np.where(clustering == max_cluster)[0], :], verbose=False)
trimmed_kemeny_rank = trimmed_kemeny_rank.astype(int)

praucs = models_performance_matrix.iloc[:, 0].to_numpy().squeeze()
f1s = models_performance_matrix.iloc[:, 1].to_numpy().squeeze()
model_names = list(models_performance_matrix.index)

print('==== Statistics ====')
print(f'Rank by PR-AUC: {rank_prauc}')
print(f'Rank by F1: {rank_f1}')
print(f'Predicted rank: {trimmed_kemeny_rank}')
print(f'Max PR-AUC: {np.max(praucs)} is achieved by {model_names[np.argmax(praucs)]}')
print(f'Max F-1: {np.max(f1s)} is achieved by {model_names[np.argmax(f1s)]}')
print(f'Our chosen model is {model_names[trimmed_kemeny_rank[0]]} which has PR-AUC={praucs[trimmed_kemeny_rank[0]]} and best F-1={f1s[trimmed_kemeny_rank[0]]}')
print(f'NDCG of predicted ranks with PR-AUC={ndcg_score(praucs.reshape((1, -1)), trimmed_kemeny_rank.reshape((1, -1)))} and best F-1={ndcg_score(f1s.reshape((1, -1)), trimmed_kemeny_rank.reshape((1, -1)))}')

