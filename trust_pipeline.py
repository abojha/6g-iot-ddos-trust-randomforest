import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_features_df(df, t_slot):
    """
    Compute the 4-dimensional feature vector K = (k1, k2, k3, k4)
    for flows of a single node in a time slot.
      k1: Activity (number of flows)
      k2: Traffic symmetry (average of sent_packets/(sent_packets+received_packets))
      k3: Average packet size
      k4: Destination entropy (of destination IP addresses)
    """
    if df.empty:
        return (0, 0.5, 0, 0)
    k1 = len(df)
    total_pkts = df['sent_packets'] + df['received_packets']
    symm = np.where(total_pkts > 0, df['sent_packets'] / total_pkts, 0.5)
    k2 = symm.mean()
    k3 = df['packet_size'].mean()
    counts = df['dst_ip'].value_counts()
    if counts.sum() > 0:
        p = counts / counts.sum()
        k4 = -(p * np.log2(p)).sum()
    else:
        k4 = 0
    return (k1, k2, k3, k4)

def compute_spatial_trust(cluster_features, node_features, theta=0.5):
    """
    Compute spatial trust by comparing node features with cluster medians.
    (Equations (1)–(3) in the paper)
    """
    features_array = np.array(cluster_features)
    medians = np.median(features_array, axis=0)
    normalized = [node_features[i] / medians[i] if medians[i] != 0 else 0.0 for i in range(4)]
    
    norm_all = []
    for feat in cluster_features:
        norm_feat = [feat[i] / medians[i] if medians[i] != 0 else 0.0 for i in range(4)]
        norm_all.append(norm_feat)
    norm_all = np.array(norm_all)
    std_devs = np.std(norm_all, axis=0, ddof=1)
    
    Z = []
    for i in range(4):
        if std_devs[i] == 0:
            Z.append(0)
        else:
            Z.append(abs(normalized[i] - 1) / std_devs[i])
    T_vals = [math.tanh(theta * z) for z in Z]
    Ts = np.mean(T_vals)
    return Ts

def compute_temporal_trust(node, current_K, history, theta=0.5):
    """
    Compute temporal trust based on historical K vectors for the node.
    (Equations (4)–(8) in the paper)
    If fewer than 3 historical samples exist, return 0.5.
    """
    if node not in history or len(history[node]) < 3:
        return 0.5
    hist = history[node]  # list of K vectors (tuples)
    n = len(hist)
    hist_arr = np.array(hist)  # shape: (n, 4)
    x = np.arange(1, n+1)
    T_values = []
    for i in range(4):
        y = hist_arr[:, i]
        slope, intercept = np.polyfit(x, y, 1)
        P_i = slope * (n + 1) + intercept
        Di = np.var(y, ddof=1) if len(y) > 1 else 0
        if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
            r_i = 0
        else:
            r_i = np.corrcoef(x, y)[0, 1]
        if n - 2 > 0 and Di > 0:
            sigma_i = math.sqrt((1 - r_i**2) / (n - 2) * Di)
        else:
            sigma_i = 0
        if sigma_i != 0:
            Z_prime = abs(current_K[i] - P_i) / sigma_i
        else:
            Z_prime = 0
        T_i = math.tanh(theta * Z_prime)
        T_values.append(T_i)
    Tt = np.mean(T_values)
    return Tt

def aggregate_trust(Ts, Tt):
    """Aggregate spatial and temporal trust using RMS (Equation (9))."""
    return math.sqrt((Ts ** 2 + Tt ** 2) / 2)

def update_node_trust(node, current_raw_trust, trust_history, slot_index, lambda_param=0.5):
    """
    Update node trust using exponential decay of historical trust values.
    (Equations (10) and (11) in the paper)
    trust_history: dictionary storing list of previous updated trust values for each node.
    slot_index: current time slot index (starting from 1).
    """
    if slot_index == 1 or node not in trust_history or len(trust_history[node]) == 0:
        # For the first slot, use the raw trust directly.
        Tk = current_raw_trust
    else:
        Q_values = [math.exp(-lambda_param * (slot_index - i)) for i in range(1, slot_index)]
        sum_Q_prev = sum(Q_values)
        Qk = 1  # exp(0) = 1 for the current slot
        Tk = (sum(trust_history[node]) + Qk * current_raw_trust) / (sum_Q_prev + Qk)
    updated_trust = 1 - Tk
    trust_history.setdefault(node, []).append(updated_trust)
    return updated_trust

def compute_flow_trust(group, node_updated_trust, node_current_K, theta=0.5, t_slot=60):
    """
    Compute flow-level trust for flows of a node.
    For each flow:
      - kf1 = t_slot / Δt, expected value 1.
      - kf2 = sent_packets/(sent_packets+received_packets), expected value 0.5.
      - kf3 = packet_size, expected value = node's average packet size (k3).
      - F4 = (Nd/Nt) * node_updated_trust, where Nd is count of flows with the same destination.
    Then raw flow trust F = 1 - (F1 + F2 + F3 + F4) / 4 (Equation (14)).
    """
    group = group.copy().sort_values(by='stime')
    group['delta_t'] = group['stime'].diff().fillna(t_slot)
    group['kf1'] = np.where(group['delta_t'] != 0, t_slot / group['delta_t'], t_slot)
    total_pkts = group['sent_packets'] + group['received_packets']
    group['kf2'] = np.where(total_pkts > 0, group['sent_packets'] / total_pkts, 0.5)
    group['kf3'] = group['packet_size']
    # F1, F2, F3: use tanh of deviation from expected values
    F1 = np.tanh(theta * np.abs(group['kf1'] - 1))
    F2 = np.tanh(theta * np.abs(group['kf2'] - 0.5))
    expected_k3 = node_current_K[2] if node_current_K[2] != 0 else 0
    F3 = np.tanh(theta * np.abs(group['kf3'] - expected_k3))
    dst_counts = group['dst_ip'].value_counts()
    group['Nd'] = group['dst_ip'].map(lambda x: dst_counts.get(x, 1))
    group['Nt'] = len(group)
    F4 = (group['Nd'] / group['Nt']) * node_updated_trust
    F = 1 - (F1 + F2 + F3 + F4) / 4
    return F




saddr_list = []
def extract_features(data, t_slot=3600, N=10, theta=0.5, lambda_param=0.5):
    # Ensure numeric conversion
    data['stime'] = pd.to_numeric(data['stime'], errors='coerce')
    data['pkts'] = pd.to_numeric(data['pkts'], errors='coerce')
    data['bytes'] = pd.to_numeric(data['bytes'], errors='coerce')
    data['spkts'] = pd.to_numeric(data['spkts'], errors='coerce')
    data['dpkts'] = pd.to_numeric(data['dpkts'], errors='coerce')

    data = data.copy()
    data['sent_packets'] = data['spkts']
    data['received_packets'] = data['dpkts']
    data['dst_ip'] = data['daddr']
    data['packet_size'] = np.where(data['pkts'] > 0, data['bytes'] / data['pkts'], 0)

    features_list = []
    labels_list = []
    time_slot_list = []  # NEW: track time_slot

    node_K_history = {}
    node_trust_history = {}

    time_slots = sorted(data['time_slot'].unique())
    time_slot_node_trust = {}
    time_slot_flow_F = {}

    for slot_idx, ts in enumerate(tqdm(time_slots, desc="Processing time slots"), start=1):
        df_ts = data[data['time_slot'] == ts].copy()
        groups = df_ts.groupby('saddr')

        node_features_dict = {}
        node_raw_trust_dict = {}
        node_updated_trust_dict = {}
        node_groups = {}
        flow_F_dict = {}

        for node, group in groups:
            group = group.sort_values(by='stime').copy()
            current_K = compute_features_df(group, t_slot)
            node_features_dict[node] = current_K
            node_K_history.setdefault(node, []).append(current_K)
            node_groups[node] = group

        cluster_features = list(node_features_dict.values())
        for node, current_K in node_features_dict.items():
            Ts_val = compute_spatial_trust(cluster_features, current_K, theta=theta)
            Tt_val = compute_temporal_trust(node, current_K, node_K_history, theta=theta)
            raw_trust = aggregate_trust(Ts_val, Tt_val)
            node_raw_trust_dict[node] = raw_trust
            updated_trust = update_node_trust(node, raw_trust, node_trust_history, slot_idx, lambda_param=lambda_param)
            node_updated_trust_dict[node] = updated_trust

        time_slot_node_trust[ts] = list(node_updated_trust_dict.values())

        for node, group in node_groups.items():
            F_series = compute_flow_trust(group, node_updated_trust_dict[node], node_features_dict[node], theta=theta, t_slot=t_slot)
            flow_F_dict[node] = F_series

        all_flow_F = pd.concat(flow_F_dict.values()) if flow_F_dict else pd.Series([])
        v_F = all_flow_F.var(ddof=1) if not all_flow_F.empty else 0
        node_trust_array = np.array(time_slot_node_trust[ts])
        v_T = np.var(node_trust_array, ddof=1) if len(node_trust_array) > 1 else 0

        final_flow_trust = {}
        for node, group in node_groups.items():
            if (v_T + v_F) != 0:
                TF = (v_F / (v_T + v_F)) * node_updated_trust_dict[node] + (v_T / (v_T + v_F)) * flow_F_dict[node]
            else:
                TF = 0.5 * (node_updated_trust_dict[node] + flow_F_dict[node])
            final_flow_trust[node] = TF

        for node, group in node_groups.items():
            k1, k2, k3, k4 = node_features_dict[node]
            node_trust = node_updated_trust_dict[node]
            tf_series = final_flow_trust[node].reset_index(drop=True)  # Align indices
            group = group.reset_index(drop=True)  # Align indices
            for i in range(len(group)):
                features_list.append([k1, k2, k3, k4, node_trust, tf_series[i]])
                time_slot_list.append(ts)
                saddr_list.append(node)
                labels_list.append(int(group.iloc[i]['attack']))




    df_features = pd.DataFrame(features_list, columns=['k1', 'k2', 'k3', 'k4', 'node_trust', 'TF'])
    df_features['attack'] = labels_list
    df_features['time_slot'] = time_slot_list  # NEW: add time_slot column
    df_features['saddr'] = saddr_list

    return df_features

