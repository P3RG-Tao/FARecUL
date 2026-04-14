import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score


def compute_neighbor(data_generator, k_hop=0):
    assert k_hop == 0
    train_data = data_generator.train.values.copy()
    matrix_size = data_generator.n_users + data_generator.n_items
    train_data[:, 1] += data_generator.n_users
    train_data[:, -1] = np.ones_like(train_data[:, -1])

    train_data2 = np.ones_like(train_data)
    train_data2[:, 0] = train_data[:, 1]
    train_data2[:, 1] = train_data[:, 0]

    padding = np.concatenate([
        np.arange(matrix_size).reshape(-1, 1),
        np.arange(matrix_size).reshape(-1, 1),
        np.ones(matrix_size).reshape(-1, 1)
    ], axis=-1)

    data = np.concatenate([train_data, train_data2, padding], axis=0).astype(int)
    train_matrix = sp.csc_matrix(
        (data[:, -1], (data[:, 0], data[:, 1])),
        shape=(matrix_size, matrix_size)
    )

    neighbor_set = []
    init_users = data_generator.train_random['user'].values.reshape(-1)
    neighbor_set.extend(np.unique(init_users))
    init_items = data_generator.train_random['item'].values.reshape(-1) + data_generator.n_users
    neighbor_set.extend(np.unique(init_items))
    neighbor_set = np.array(neighbor_set)

    users_nei = neighbor_set[np.where(neighbor_set < data_generator.n_users)]
    items_nei = neighbor_set[np.where(neighbor_set >= data_generator.n_users)] - data_generator.n_users

    return users_nei, items_nei


def get_eval_mask(data_generator):
    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values

    nei_users, nei_items = compute_neighbor(data_generator)
    nei_users = set(nei_users.tolist())
    nei_items = set(nei_items.tolist())

    # mask or for valid
    mask_1 = np.zeros(valid_data.shape[0], dtype=np.int64)
    for ii in range(valid_data.shape[0]):
        u, i = valid_data[ii, 0], valid_data[ii, 1]
        if u in nei_users or i in nei_items:
            mask_1[ii] = 1
    mask_1 = np.where(mask_1 > 0)[0]

    # mask or for test
    mask_2 = np.zeros(test_data.shape[0], dtype=np.int64)
    for ii in range(test_data.shape[0]):
        u, i = test_data[ii, 0], test_data[ii, 1]
        if u in nei_users or i in nei_items:
            mask_2[ii] = 1
    mask_2 = np.where(mask_2 > 0)[0]

    # mask and for valid
    mask_3 = np.zeros(valid_data.shape[0], dtype=np.int64)
    for ii in range(valid_data.shape[0]):
        u, i = valid_data[ii, 0], valid_data[ii, 1]
        if u in nei_users and i in nei_items:
            mask_3[ii] = 1
    mask_3 = np.where(mask_3 > 0)[0]

    # mask and for test
    mask_4 = np.zeros(test_data.shape[0], dtype=np.int64)
    for ii in range(test_data.shape[0]):
        u, i = test_data[ii, 0], test_data[ii, 1]
        if u in nei_users and i in nei_items:
            mask_4[ii] = 1
    mask_4 = np.where(mask_4 > 0)[0]

    return mask_1, mask_2, mask_3, mask_4


def safe_auc(y_true, y_score):
    """
    安全计算AUC：
    若样本数不足，或标签只有一个类别，则返回0.0，避免报错
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(y_true) == 0:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_score)


def _compute_topn_per_user(user_ids, item_ids, true_labels, predictions, top_k_list):
    """
    按用户计算 Top-N 指标:
    Recall@K, NDCG@K
    """
    user_ids = np.asarray(user_ids).reshape(-1)
    item_ids = np.asarray(item_ids).reshape(-1)
    true_labels = np.asarray(true_labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    metrics = {}
    for k in top_k_list:
        metrics[f'Recall@{k}'] = 0.0
        metrics[f'NDCG@{k}'] = 0.0

    unique_users = np.unique(user_ids)
    user_count = 0

    for user in unique_users:
        user_mask = (user_ids == user)

        user_true = true_labels[user_mask]
        user_preds = predictions[user_mask]

        pos_count = np.sum(user_true == 1)
        neg_count = np.sum(user_true == 0)
        if pos_count == 0 or neg_count == 0:
            continue

        sorted_idx = np.argsort(user_preds)[::-1]
        sorted_true = user_true[sorted_idx]

        user_count += 1

        for k in top_k_list:
            effective_k = min(k, len(sorted_true))
            if effective_k == 0:
                continue

            top_k_true = sorted_true[:effective_k]
            hit = np.sum(top_k_true == 1)

            recall = hit / pos_count

            dcg = 0.0
            for idx, label in enumerate(top_k_true):
                if label == 1:
                    dcg += 1.0 / np.log2(idx + 2)

            ideal_k = min(effective_k, pos_count)
            idcg = 0.0
            for idx in range(ideal_k):
                idcg += 1.0 / np.log2(idx + 2)

            ndcg = dcg / idcg if idcg > 0 else 0.0

            metrics[f'Recall@{k}'] += recall
            metrics[f'NDCG@{k}'] += ndcg

    if user_count > 0:
        for k in top_k_list:
            metrics[f'Recall@{k}'] /= user_count
            metrics[f'NDCG@{k}'] /= user_count
    else:
        for k in top_k_list:
            metrics[f'Recall@{k}'] = 0.0
            metrics[f'NDCG@{k}'] = 0.0

    return metrics


def calculate_topn_metrics(data_generator, user_ids, item_ids, true_labels, predictions, top_k_list=[10, 20]):
    """
    计算整体 / OR / AND 三类 Top-N 指标
    """
    user_ids = np.asarray(user_ids).reshape(-1)
    item_ids = np.asarray(item_ids).reshape(-1)
    true_labels = np.asarray(true_labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    metrics = {
        'overall': {},
        'or': {},
        'and': {}
    }

    # 整体
    overall_metrics = _compute_topn_per_user(
        user_ids, item_ids, true_labels, predictions, top_k_list
    )
    metrics['overall'] = overall_metrics

    # OR / AND 子集
    nei_users, nei_items = compute_neighbor(data_generator)
    nei_users = set(nei_users.tolist())
    nei_items = set(nei_items.tolist())

    mask_or = np.array(
        [1 if (u in nei_users or i in nei_items) else 0 for u, i in zip(user_ids, item_ids)],
        dtype=np.int64
    )
    mask_and = np.array(
        [1 if (u in nei_users and i in nei_items) else 0 for u, i in zip(user_ids, item_ids)],
        dtype=np.int64
    )

    or_idx = np.where(mask_or == 1)[0]
    and_idx = np.where(mask_and == 1)[0]

    if len(or_idx) > 0:
        metrics['or'] = _compute_topn_per_user(
            user_ids[or_idx],
            item_ids[or_idx],
            true_labels[or_idx],
            predictions[or_idx],
            top_k_list
        )
    else:
        for k in top_k_list:
            metrics['or'][f'Recall@{k}'] = 0.0
            metrics['or'][f'NDCG@{k}'] = 0.0

    if len(and_idx) > 0:
        metrics['and'] = _compute_topn_per_user(
            user_ids[and_idx],
            item_ids[and_idx],
            true_labels[and_idx],
            predictions[and_idx],
            top_k_list
        )
    else:
        for k in top_k_list:
            metrics['and'][f'Recall@{k}'] = 0.0
            metrics['and'][f'NDCG@{k}'] = 0.0

    return metrics


def format_topn_metrics(topn_metrics, top_k_list):

    parts = []

    for scope in ['overall', 'or', 'and']:
        scope_parts = [f'[{scope}]']
        for k in top_k_list:
            scope_parts.append(
                f'R@{k}={topn_metrics[scope][f"Recall@{k}"]:.4f}, '
                f'N@{k}={topn_metrics[scope][f"NDCG@{k}"]:.4f}'
            )
        parts.append(' '.join(scope_parts))

    return ' | '.join(parts)


def get_eval_result(data_generator, model, mask, top_k_list=[10, 20]):
    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values

    with torch.no_grad():
        valid_predictions = model.predict(valid_data[:, 0], valid_data[:, 1])
        if isinstance(valid_predictions, torch.Tensor):
            valid_predictions = valid_predictions.detach().cpu().numpy()

        test_predictions = model.predict(test_data[:, 0], test_data[:, 1])
        if isinstance(test_predictions, torch.Tensor):
            test_predictions = test_predictions.detach().cpu().numpy()

    valid_predictions = np.asarray(valid_predictions).reshape(-1)
    test_predictions = np.asarray(test_predictions).reshape(-1)

    # AUC
    valid_auc = safe_auc(valid_data[:, -1], valid_predictions)
    valid_auc_or = safe_auc(valid_data[:, -1][mask[0]], valid_predictions[mask[0]])
    valid_auc_and = safe_auc(valid_data[:, -1][mask[2]], valid_predictions[mask[2]])

    test_auc = safe_auc(test_data[:, -1], test_predictions)
    test_auc_or = safe_auc(test_data[:, -1][mask[1]], test_predictions[mask[1]])
    test_auc_and = safe_auc(test_data[:, -1][mask[3]], test_predictions[mask[3]])

    # Top-N
    valid_topn = calculate_topn_metrics(
        data_generator,
        valid_data[:, 0],
        valid_data[:, 1],
        valid_data[:, 2],
        valid_predictions,
        top_k_list
    )
    test_topn = calculate_topn_metrics(
        data_generator,
        test_data[:, 0],
        test_data[:, 1],
        test_data[:, 2],
        test_predictions,
        top_k_list
    )

    return (
        valid_auc, valid_auc_or, valid_auc_and,
        test_auc, test_auc_or, test_auc_and,
        valid_topn, test_topn
    )


def get_eval_result_original(data_generator, model, mask):
    results = get_eval_result(data_generator, model, mask, top_k_list=[10, 20])
    return results[:6]