import os
import time
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from concurrent.futures import ProcessPoolExecutor, as_completed
import json


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder
import pickle



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. 定义推理专用 Dataset
class InferenceDataset(Dataset):
    def __init__(self, uuids, llm_vector_dict, user_feature_dict):
        # 仅保留两边都有数据的有效用户
        self.uuids = [u for u in uuids if u in llm_vector_dict and u in user_feature_dict]
        self.llm_vectors = [llm_vector_dict[u] for u in self.uuids]
        self.user_features = [user_feature_dict[u] for u in self.uuids]

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        return (
            self.uuids[idx],
            torch.tensor(self.llm_vectors[idx], dtype=torch.float32),
            torch.tensor(self.user_features[idx], dtype=torch.float32)
        )

# 2. 批量推理函数
def batch_generate_aligned_values(model, uuids, llm_vector_dict, user_feature_dict, batch_size=256):
    """
    输入：训练好的 model，所有用户的 uuid 列表，LLM 向量字典，行为特征字典
    输出：包含所有用户对齐后价值向量的字典 {uuid: aligned_vector_numpy}
    """
    model.eval() # 切换到评估模式（关闭 Dropout 等）
    
    # 自动识别模型所在设备 (CPU / GPU)
    device = next(model.parameters()).device
    
    dataset = InferenceDataset(uuids, llm_vector_dict, user_feature_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    aligned_results = {}
    
    print(f"开始批量生成对齐价值向量，总用户数: {len(dataset)} ...")
    
    with torch.no_grad(): # 推理阶段不需要计算梯度，省内存提速
        for batch_uuids, llm_vecs, user_fs in dataloader:
            llm_vecs = llm_vecs.to(device)
            user_fs = user_fs.to(device)
            
            # 前向传播：获取对齐后的价值向量 v_aligned
            v_aligned, delta_v = model(llm_vecs, user_fs)
            
            # 转回 NumPy 格式
            v_aligned_np = v_aligned.cpu().numpy()
            
            # 存入字典
            for uuid, vec in zip(batch_uuids, v_aligned_np):
                aligned_results[uuid] = vec
                
    print("所有用户的价值向量对齐与导出完成！")
    return aligned_results

# ==========================================
# 调用示例
# ==========================================
# 假设 model 是训练好的模型实例，uuids 是全部 10 万用户的 uuid 列表
# aligned_vectors_dict = batch_generate_aligned_values(model, uuids, llm_vector_dict, user_comprehensive_features_dict)

# # 示例：查看某个用户的对齐后价值向量
# sample_uuid = uuids[0]
# print(f"用户 {sample_uuid} 对齐后的价值向量形状: {aligned_vectors_dict[sample_uuid].shape}")





# ==========================================
# 0. 基础几何与特征编码工具函数
# ==========================================
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0 # 地球半径 (km)
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def encode_poi_features(df_pois, category_encoder):
    """
    将 POI 记录转化为数值特征向量 phi(p)
    """
    # 类别特征编码
    cat_features = category_encoder.transform(df_pois[['first_category_name', 'third_category_name']])
    
    # 空间与数值特征归一化
    num_features = np.stack([
        df_pois['lnt_wgs84'].values / 180.0,
        df_pois['lat_wgs84'].values / 90.0,
        np.log1p(df_pois['stay_minutes'].values) / 10.0
    ], axis=-1)
    
    phi_p = np.hstack([cat_features, num_features])
    return phi_p.astype(np.float32)


# ==========================================
# 1. 单个用户特征与样本提取函数（供多进程并行调用）
# ==========================================
# (uuid, group, global_poi_pool, category_encoder, K) is args
def process_single_user(args):
    """
    worker 函数：处理单个用户的全部轨迹，提取 f_u 并构建 (chosen, rejected) 训练对
    """
    uuid, user_df, global_poi_pool, category_encoder, K = args
    
    # 1. 提取用户宏观行为特征 f_u
    lons = user_df['lnt_wgs84'].values
    lats = user_df['lat_wgs84'].values
    center_lon, center_lat = np.mean(lons), np.mean(lats)
    distances = [haversine(lons[i], lats[i], center_lon, center_lat) for i in range(len(lons))]
    
    night_visits = user_df[(user_df['start_hour'] >= 22) | (user_df['start_hour'] <= 6)]
    
    f_u_dict = {
        'radius_of_gyration': np.mean(distances) if distances else 0.0,
        'max_distance': np.max(distances) if distances else 0.0,
        'night_ratio': len(night_visits) / len(user_df) if len(user_df) > 0 else 0.0,
        'mean_stay_minutes': user_df['stay_minutes'].mean(),
        'stay_minutes_std': user_df['stay_minutes'].std() if len(user_df) > 1 else 0.0,
        'total_visits': len(user_df),
        'active_days': user_df['date_str'].nunique(),
        'hotel_visit_ratio': len(user_df[user_df['first_category_name'] == '酒店']) / len(user_df) if len(user_df) > 0 else 0.0
    }
    f_u = np.array(list(f_u_dict.values()), dtype=np.float32)
    
    # 2. 构建正样本与硬负样本对
    user_samples = []
    chosen_records = user_df[user_df['stay_minutes'] > 120]
    if len(chosen_records) == 0:
        chosen_records = user_df
        
    user_visited_poi_ids = set(user_df['poi_id'].unique())
    
    for _, row in chosen_records.iterrows():
        chosen_poi_df = pd.DataFrame([row])
        phi_chosen = encode_poi_features(chosen_poi_df, category_encoder)[0]
        
        target_c_code = row['c_code']
        target_third_cat = row['third_category_name']
        cur_lon, cur_lat = row['lnt_wgs84'], row['lat_wgs84']
        
        # 筛选硬负例候选池
        candidates = global_poi_pool[
            (global_poi_pool['c_code'] == target_c_code) & 
            (global_poi_pool['third_category_name'] == target_third_cat) & 
            (~global_poi_pool['poi_id'].isin(user_visited_poi_ids))
        ].copy()
        
        if len(candidates) > 0:
            candidates['dist'] = candidates.apply(
                lambda r: haversine(cur_lon, cur_lat, r['lnt_wgs84'], r['lat_wgs84']), axis=1
            )
            # 距离单位是公里
            hard_negatives = candidates[candidates['dist'] <= 5.0]
            
            sampled_negs = None
            if len(hard_negatives) >= K:
                sampled_negs = hard_negatives.sample(n=K)
            elif len(candidates) >= K:
                sampled_negs = candidates.sample(n=K)
                
            if sampled_negs is not None:
                phi_rejected = encode_poi_features(sampled_negs, category_encoder) # [K, dim_phi]
                user_samples.append({
                    'f_u': f_u,
                    'phi_chosen': phi_chosen,
                    'phi_rejected': phi_rejected
                })
                
    return uuid, user_samples, f_u


# ==========================================
# 2. PyTorch Dataset 定义
# ==========================================
class ResidualDataset(Dataset):
    def __init__(self, processed_samples, llm_vector_dict):
        """
        processed_samples: 包含 (uuid, f_u, phi_chosen, phi_rejected) 的平铺样本列表
        llm_vector_dict: {uuid: llm_vector_array} 映射字典
        """
        self.samples = []
        for sample in processed_samples:
            uuid = sample['uuid']
            if uuid in llm_vector_dict:
                self.samples.append({
                    'llm_vector': llm_vector_dict[uuid],
                    'f_u': sample['f_u'],
                    'phi_chosen': sample['phi_chosen'],
                    'phi_rejected': sample['phi_rejected']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.tensor(item['llm_vector'], dtype=torch.float32),
            torch.tensor(item['f_u'], dtype=torch.float32),
            torch.tensor(item['phi_chosen'], dtype=torch.float32),
            torch.tensor(item['phi_rejected'], dtype=torch.float32)
        )


# ==========================================
# 3. 残差修正网络与损失函数
# ==========================================
class ResidualCorrectionNetwork(nn.Module):
    def __init__(self, dim_v, dim_f, hidden_dim=128, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Linear(dim_v + dim_f, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_v)
        )
        self.activation = nn.Tanh()

    def forward(self, llm_vector, user_feature):
        combined = torch.cat([llm_vector, user_feature], dim=-1)
        delta_v = self.alpha * self.activation(self.mlp(combined))
        
        v_unbounded = llm_vector + delta_v
        
        v_aligned = torch.clamp(v_unbounded, min=-1.0, max=1.0)
        
        # v_aligned = llm_vector + delta_v
        return v_aligned, delta_v


class ResidualMLPBlock(nn.Module):
    """
    残差 MLP 基础块：通过跳跃连接 (x + block(x)) 增强非线性特征表达，同时防止梯度消失
    """
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 层归一化，稳定中小数据集的训练
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)  # 残差相加


# class ResidualCorrectionNetwork(nn.Module):
#     """
#     集成 ResidualMLPBlock 的残差修正网络
#     """
#     def __init__(self, dim_v, dim_f, hidden_dim=128, num_blocks=2, alpha=0.1, dropout=0.1):
#         super().__init__()
#         self.alpha = alpha
        
#         # 1. 输入投影层：将拼接后的异构特征映射到统一的隐藏维度
#         self.input_proj = nn.Sequential(
#             nn.Linear(dim_v + dim_f, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # 2. 核心特征交互层：动态堆叠指定数量（如 2 个）的残差块
#         self.res_blocks = nn.ModuleList([
#             ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout) 
#             for _ in range(num_blocks)
#         ])
        
#         # 3. 输出投影层：将隐藏特征映射回 88 维的价值增量空间
#         self.output_proj = nn.Linear(hidden_dim, dim_v)
#         self.activation = nn.Tanh()

#     def forward(self, llm_vector, user_feature):
#         # 拼接 LLM 初始价值向量与用户宏观行为特征
#         combined = torch.cat([llm_vector, user_feature], dim=-1)
        
#         # 特征投影
#         x = self.input_proj(combined)
        
#         # 依次通过残差块进行深层特征交互
#         for block in self.res_blocks:
#             x = block(x)
            
#         # 计算价值增量 delta_v 并通过缩放与 Tanh 限制幅度
#         delta_v = self.alpha * self.activation(self.output_proj(x))
        
#         # 基础加和
#         v_unbounded = llm_vector + delta_v
        
#         # 严格限制在 [-1.0, 1.0] 区间内
#         v_aligned = torch.clamp(v_unbounded, min=-1.0, max=1.0)
        
#         return v_aligned, delta_v




class InfoNCELossWithRegularization(nn.Module):
    def __init__(self, dim_phi, dim_v, temperature=0.07, beta=0.01):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        # 投影层：将 POI 特征维度投影到与价值向量一致的维度
        self.poi_projection = nn.Linear(dim_phi, dim_v)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, v_aligned, chosen_poi, rejected_pois, delta_v):
        # 投影 POI 特征
        proj_chosen = self.poi_projection(chosen_poi) # [Batch, dim_v]
        proj_rejected = self.poi_projection(rejected_pois) # [Batch, K, dim_v]

        # 计算得分
        pos_score = torch.sum(v_aligned * proj_chosen, dim=-1) / self.temperature
        neg_score = torch.sum(v_aligned.unsqueeze(1) * proj_rejected, dim=-1) / self.temperature

        logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss_infonce = self.criterion(logits, labels)
        loss_reg = torch.mean(torch.sum(delta_v ** 2, dim=-1))
        
        return loss_infonce + self.beta * loss_reg


# ==========================================
# 4. 主程序执行入口（端到端流水线）
# ==========================================
if __name__ == "__main__":
    print("=== Step 1: 模拟生成海量 POI 数据与 LLM 价值向量 ===" )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(42)
    num_users = 988  # 演示用 500 人，实际可设为 100,000
    K = 5            # 每个正样本的硬负样本数
    dim_v = 88       # LLM 价值向量维度

    # # 模拟生成原始 POI 日志列表
    # mock_logs = []
    # uuids = [f"user_{i:05d}" for i in range(num_users)]
    
    # for uuid in uuids:
    #     num_records = np.random.randint(10, 30)
    #     for _ in range(num_records):
    #         mock_logs.append({
    #             'uuid': uuid,
    #             'poi_id': np.random.randint(10000, 99999),
    #             'start_hour': np.random.randint(0, 24),
    #             'end_hour': np.random.randint(0, 24),
    #             'start_time': 1730000000000 + np.random.randint(0, 86400000),
    #             'end_time': 1730000000000 + np.random.randint(86400000, 172800000),
    #             'location': '114.53,30.48',
    #             'stay_minutes': np.random.randint(30, 300),
    #             'source': 'wifi_appearance',
    #             'c_code': 420100,
    #             'lnt_wgs84': 114.5 + np.random.randn() * 0.05,
    #             'lat_wgs84': 30.4 + np.random.randn() * 0.05,
    #             'date_str': '2024-10-27',
    #             'first_category_name': np.random.choice(['酒店', '餐饮', '购物']),
    #             'third_category_name': np.random.choice(['经济酒店', '中端餐饮', '便利店'])
    #         })

    # df_logs = pd.DataFrame(mock_logs)
    # global_poi_pool = df_logs[['poi_id', 'c_code', 'lnt_wgs84', 'lat_wgs84', 'first_category_name', 'third_category_name']].drop_duplicates().reset_index(drop=True)

    
    clean_file = './clean_data_with_category_delete_none_988.csv'
    df = pd.read_csv(clean_file)
    
    global_poi_pool = df[['poi_id', 'stay_minutes', 'c_code', 'lnt_wgs84', 'lat_wgs84', 'first_category_name', 'third_category_name']].drop_duplicates().reset_index(drop=True)
    value_file = './person_value_vector_988.jsonl'

    uuid_list = []
    uuid_dict = {}
    with open(value_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 确保行不为空
                data = json.loads(line.strip())
                uuid_list.append(list(data.keys())[0])  # 获取每行的第一个键，即uuid
                uuid_dict[list(data.keys())[0]] = data[list(data.keys())[0]]
                
    # print(uuid_list)
    print(len(uuid_list))
    # uuid_set = set(uuid_list)
    
    
    
    # 拟合全局类别编码器
    category_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    # category_encoder.fit(global_poi_pool[['first_category_name', 'third_category_name']])
    
    poi_df = pd.read_csv('./_ods_bd_poi__202506201108.csv')
    category_encoder.fit(poi_df[['first_category_name', 'third_category_name']])

    # 模拟 LLM 生成的初始价值向量字典 {uuid: vector}
    # llm_vector_dict = {uuid: np.random.randn(dim_v) for uuid in uuids}
    llm_vector_dict = uuid_dict  # 使用实际的 LLM 价值向量数据


    print("\n=== Step 2: 多进程并行提取特征与构建训练样本 ===" )
    start_time = time.time()
    
    # 准备多进程传参：按用户分组
    # user_groups = [(uuid, group, global_poi_pool, category_encoder, K) for uuid, group in df.groupby('uuid')]
    
    # processed_samples = []
    # user_comprehensive_features_dict = {}
    # # 使用多进程并发处理（充分利用多核 CPU）
    # max_workers = os.cpu_count() or 4
    # print(f"启动多进程池，使用 worker 数量: {max_workers}")
    
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(process_single_user, arg) for arg in user_groups]
    #     for future in as_completed(futures):
    #         uuid, user_samples, f_u = future.result()
            
    #         user_comprehensive_features_dict[uuid] = f_u
    #         for sample in user_samples:
    #             sample['uuid'] = uuid
    #             processed_samples.append(sample)

    # print(f"特征工程与硬负采样完成！耗时: {time.time() - start_time:.2f} 秒。有效训练对总数: {len(processed_samples)}")
    

    
    # f3 = open("processed_samples.pkl", "wb")
    # pickle.dump(processed_samples, f3)
    # f3.close()
    
    
    # with open("user_comprehensive_features_dict.pkl", "wb") as f:
    #     pickle.dump(user_comprehensive_features_dict, f)
    # print("用户宏观行为特征已成功缓存至本地！")
    
    with open("processed_samples.pkl", "rb") as f:
        processed_samples = pickle.load(f)
    
    with open("user_comprehensive_features_dict.pkl", "rb") as f:
        user_comprehensive_features_dict = pickle.load(f)



    print("\n=== Step 3: 构建 PyTorch Dataset 与 DataLoader ===" )
    dataset = ResidualDataset(processed_samples, llm_vector_dict)
    if len(dataset) == 0:
        raise ValueError("训练样本为空，请检查数据过滤条件！")
        
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 动态获取特征维度
    sample_llm, sample_f_u, sample_chosen, sample_rejected = dataset[0]
    dim_f = sample_f_u.shape[0]
    dim_phi = sample_chosen.shape[0]

    # 初始化模型、损失函数与优化器
    # (self, dim_v, dim_f, hidden_dim=128, num_blocks=2, alpha=0.1, dropout=0.1)
    model = ResidualCorrectionNetwork(dim_v=dim_v, dim_f=dim_f, hidden_dim=128, alpha=0.1).to(device)
    criterion = InfoNCELossWithRegularization(dim_phi=dim_phi, dim_v=dim_v, temperature=0.07, beta=0.001).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=1e-3)


    print("\n=== Step 4: 开始训练残差修正网络 ===" )
    epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    save_directory = 'models'
    os.makedirs(save_directory, exist_ok=True)
    save_model_path = os.path.join(save_directory, "residual_correction_model_cosine_learning_rate_resnet.pth")
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # llm_vecs, user_fs, chosen_pois, rejected_pois = batch
            llm_vecs, user_fs, chosen_pois, rejected_pois = [item.to(device) for item in batch]
            # print(llm_vecs.shape, user_fs.shape, chosen_pois.shape, rejected_pois.shape)
            
            optimizer.zero_grad()
            
            # 前向传播
            v_aligned, delta_v = model(llm_vecs, user_fs)
            
            # 计算 InfoNCE 对比损失 + 正则化
            loss = criterion(v_aligned, chosen_pois, rejected_pois, delta_v)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # print(f"Epoch [{epoch+1}/{epochs}], 平均训练 Loss: {epoch_loss / len(dataloader):.4f}")
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}], 平均训练 Loss: {epoch_loss / len(dataloader):.4f}, 当前学习率: {current_lr:.6f}")

    torch.save(model.state_dict(), save_model_path)
    
    print("\n=== 训练全部顺利完成！您可以将训练好的残差模型用于对齐 10 万用户的价值向量。 ===" )
    
    
    # 假设 model 是训练好的模型实例，uuids 是全部 10 万用户的 uuid 列表
    aligned_vectors_dict = batch_generate_aligned_values(model, list(user_comprehensive_features_dict.keys()), llm_vector_dict, user_comprehensive_features_dict)

    # # 示例：查看某个用户的对齐后价值向量
    sample_uuid = uuid_list[0]
    print(aligned_vectors_dict[sample_uuid])
    # print(f"用户 {sample_uuid} 对齐后的价值向量形状: {aligned_vectors_dict[sample_uuid].shape}")