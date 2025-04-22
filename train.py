#这是划分好数据集之后，在train_dataset数据集上训练，包含混合精度的，但是可能在train_epoch有点问题，又不是完全混合精度
import os, json, random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ========= 投影层 ==========
class ProjectionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.activation(x)

def augment_data(item):
    """增强策略：代码变量名替换 + 文本同义词替换"""
    # 随机遮盖代码中的变量名
    code_tokens = item['bug_cpp_code'].split()
    var_dict = {}
    masked_code = []
    for token in code_tokens:
        if token.isidentifier() and not token[0].isdigit():
            if token not in var_dict:
                var_dict[token] = f"var_{len(var_dict)+1}"
            masked_code.append(var_dict[token])
        else:
            masked_code.append(token)
    
    # 文本同义词替换
    text = item['bug_text'].lower()
    synonym_map = {
        'error': ['issue', 'fault', 'problem'],
        'fix': ['correct', 'repair', 'resolve'],
        'bug': ['defect', 'flaw', 'mistake']
    }
    for word, replacements in synonym_map.items():
        if word in text:
            text = text.replace(word, random.choice(replacements))
    
    return {
        "bug_text": text,
        "bug_cpp_code": " ".join(masked_code),
        "code_text": item["code_text"],
        "label": item["label"]
    }
    
class OptimizedHierarchicalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.token_attn = nn.Linear(hidden_size, 1)  # 单层感知机替代多头注意力
        self.chunk_attn = nn.Linear(hidden_size, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, chunk_vectors):

        # Token级注意力
        token_weights = F.softmax(self.token_attn(chunk_vectors), dim=1)  # [B, num_chunks, 1]
        token_out = torch.sum(token_weights * chunk_vectors, dim=1)  # [B, H]
        
        # Chunk级注意力
        chunk_weights = F.softmax(self.chunk_attn(chunk_vectors.mean(dim=1)), dim=0)  # [B, 1]
        chunk_out = torch.sum(chunk_weights.unsqueeze(-1) * chunk_vectors, dim=1)  # [B, H]
        
        # 动态融合
        return self.alpha * token_out + (1 - self.alpha) * chunk_out
# ========= 编码器 ==========
class EnhancedEncoder(nn.Module):
    def __init__(self, tokenizer, model, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.projection = ProjectionHead(model.config.hidden_size).to(device)
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 注册为参数
        self.attn_query = nn.Parameter(torch.randn(model.config.hidden_size).to(device))
         #交叉注意力融合层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model.config.hidden_size,
            num_heads=8,
            batch_first=True
        ).to(device)
        self.fusion_gate = nn.Sequential(
            nn.Linear(model.config.hidden_size*2, 1),
            nn.Sigmoid()
        ).to(device)
        self.hier_attn = OptimizedHierarchicalAttention(model.config.hidden_size).to(device)

    def fuse_features(self, bug_vec, code_vec):
        # 交叉注意力
        bug_vec = F.normalize(bug_vec, p=2, dim=-1)
        code_vec = F.normalize(code_vec, p=2, dim=-1)
        attn_out, _ = self.cross_attn(
            query=bug_vec.unsqueeze(1),
            key=code_vec.unsqueeze(1),
            value=code_vec.unsqueeze(1)
        )
        attn_out = attn_out.squeeze(1)
        
        # 门控融合
        gate = self.fusion_gate(torch.cat([bug_vec, code_vec], dim=-1))
        return gate * attn_out + (1 - gate) * bug_vec

    def chunk_encode(self, text_list, max_length=512):
        all_vecs = []
        for text in text_list:
            with torch.no_grad(), torch.cuda.amp.autocast():

                inputs = self.tokenizer(text,padding=False,truncation=False,return_tensors="pt",add_special_tokens=True).to(self.device)
            chunks = []
            for i in range(0, len(inputs['input_ids'][0]), max_length - 2):
                chunk = {k: v[:, i:i + max_length] for k, v in inputs.items()}
                chunks.append(chunk)
            
            chunk_vecs = []
            for chunk in chunks:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    outputs = self.model(**chunk)
                    cls_vec = outputs.last_hidden_state[:, 0, :]
                    projected = self.projection(cls_vec)
                    chunk_vecs.append(projected)
                del outputs, cls_vec, projected
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if chunk_vecs:

                # 层次化注意力处理
                with torch.no_grad(), torch.cuda.amp.autocast():
                    chunk_vectors = torch.stack(chunk_vecs, dim=1)  # [B, num_chunks, H]
                    aggregated = self.hier_attn(chunk_vectors)  # [B, H]
                    all_vecs.append(aggregated.squeeze(0))
                del chunk_vectors, chunk_vecs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                all_vecs.append(torch.zeros(self.model.config.hidden_size,dtype=torch.float16 ).to(self.device))
        
        return torch.stack(all_vecs)
        

# ========= 数据集 ==========
class DualTowerDataset(Dataset):
    def __init__(self, data, is_train=False):
        self.data = data
        self.is_train = is_train #仅在训练模式数据增强
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.is_train and random.random() < 0.5:  # 50%概率应用增强
            return augment_data(item)
        return item

def collate_fn(batch):
    return batch


# ========= 日志记录器 ==========
class ResultLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        self.results = []
        self.detailed_predictions = []
        
    def add(self, epoch, metrics,detailed=None):
        record = {
            "epoch": epoch,
            "train_loss": metrics["train_loss"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "threshold": metrics["threshold"],
            "confusion_matrix": {
                "TP": int(metrics["TP"]),
                "TN": int(metrics["TN"]),
                "FP": int(metrics["FP"]),
                "FN": int(metrics["FN"])
            }
        }
        self.results.append(record)
        if detailed:
            self.detailed_predictions.append({
                "epoch": epoch,
                "predictions": detailed
            })
        
    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ 评估结果已保存至 {self.save_path}")
        # 保存详细预测结果
        detailed_path = os.path.splitext(self.save_path)[0] + "_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(self.detailed_predictions, f, indent=2)
        print(f"✅ 详细预测结果已保存至 {detailed_path}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """ 使用二元logits的Focal Loss """
        # 自动包含sigmoid的稳定计算
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 正确分类的概率
        loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return loss.mean()

focal_loss = FocalLoss()  # 在训练开始前初始化损失函数

class HardMiningLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, pos_sim, neg_sim):
        # 选择困难负样本
        hard_neg = torch.relu(neg_sim - self.margin + pos_sim.unsqueeze(1))
        return torch.mean(torch.log(1 + torch.exp(hard_neg)))

# ========= 训练函数 ==========

def train_epoch(encoder, data_loader, optimizer, device):
    encoder.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        # 直接不使用混合精度，正常计算
        code_texts = [item["code_text"] for item in batch]
        bug_texts = [item['bug_text'] for item in batch]
        br_cpp_texts = [item['bug_cpp_code'] for item in batch]
        
        # 使用内存优化后的编码方法
        code_vecs = encoder.chunk_encode(code_texts)  
        bug_vecs = encoder.chunk_encode(bug_texts)
        cpp_vecs = encoder.chunk_encode(br_cpp_texts)
        
        # 融合特征
        combined_vecs = encoder.fuse_features(bug_vecs, cpp_vecs)
        
        # 相似度计算
        pos_sim = torch.cosine_similarity(combined_vecs, code_vecs, dim=-1)
        neg_sim = 1 - torch.abs(combined_vecs.unsqueeze(1) - code_vecs.unsqueeze(0)).mean(dim=-1)
        neg_sim = neg_sim[~torch.eye(len(batch), dtype=bool)].view(len(batch), -1)
        
        # 概率计算
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        probs = F.softmax(logits, dim=1)[:, 0]
        
        # 损失计算
        labels = torch.ones(len(batch), dtype=torch.float32, device=device)
        loss = focal_loss(probs, labels) + 0.3 * HardMiningLoss()(pos_sim, neg_sim)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item()
        torch.cuda.empty_cache()  # 每步清理显存
    
    return total_loss / len(data_loader)

    
#动态阈值调整
def find_optimal_threshold(scores, labels):
    thresholds = np.linspace(0.0, 1.0, 101)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"🔍 最佳阈值: {best_thresh:.2f} | 对应F1: {best_f1:.4f}")
    return best_thresh

# ========= 评估函数 ==========
def evaluate(encoder, data_loader, device):
    encoder.eval()
    preds, gts, all_scores = [], [], []
    all_scores = []
    detailed = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            code_texts = [item["code_text"] for item in batch]

            bug_texts = [item['bug_text'] for item in batch]
            br_cpp_texts = [item['bug_cpp_code'] for item in batch]
            bug_vecs = encoder.chunk_encode(bug_texts)
            cpp_vecs = encoder.chunk_encode(br_cpp_texts)
            combined_vecs = encoder.fuse_features(bug_vecs, cpp_vecs)  # 使用融合层
            
            code_vecs = encoder.chunk_encode(code_texts)
            sim_scores = []
            for i in range(len(batch)):
        # 逐样本计算正确配对
                sim = torch.cosine_similarity(combined_vecs[i], code_vecs[i], dim=0)
                sim_scores.append(sim.item())
            sim_scores = np.array(sim_scores)
            batch_gts = [item["label"] for item in batch]
            
            preds.extend(sim_scores)
            gts.extend(batch_gts)
            all_scores.extend(sim_scores)
            # 记录详细预测信息
            for idx in range(len(batch)):
                detailed.append({
                    "bug_text": batch[idx]["bug_text"],
                    "code_text": batch[idx]["code_text"],
                    "score": float(sim_scores[idx]),
                    "true_label": int(batch[idx]["label"])
                })

    
    # 在所有分数上找最佳阈值
    optimal_thresh = find_optimal_threshold(np.array(all_scores), np.array(gts))
    final_preds = (np.array(all_scores) > optimal_thresh).astype(int)
    # 计算评估指标
    y_true = np.array(gts)
    
    TP = np.sum((final_preds == 1) & (y_true == 1))
    TN = np.sum((final_preds == 0) & (y_true == 0))
    FP = np.sum((final_preds == 1) & (y_true == 0))
    FN = np.sum((final_preds == 0) & (y_true == 1))
    
    acc = accuracy_score(gts, final_preds)
    p, r, f1, _ = precision_recall_fscore_support(gts, final_preds, average='binary')
    return acc, p, r, f1, optimal_thresh,TP, TN, FP, FN, detailed

def load_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f]
   
# ========= 主程序入口 ==========
if __name__ == "__main__":

    train_data = load_jsonl("train_data.jsonl")
    test_data = load_jsonl("valid_data.jsonl")
    def print_distribution(name, dataset):
        pos = sum(1 for x in dataset if x["label"] == 1)
        print(f"{name}: {len(dataset)}条 | 正样本: {pos}({pos / len(dataset):.1%})")
    
    print_distribution("训练集", train_data)
    print_distribution("测试集", test_data)
    
   
    train_labels = [x["label"] for x in train_data]
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[train_labels]
    
    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_data) * 5,
        replacement=True
    )
    
    train_loader = DataLoader(
        DualTowerDataset(train_data,is_train=True),#启用数据增强
        batch_size=2,
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        DualTowerDataset(test_data),
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    tokenizer = AutoTokenizer.from_pretrained("/media/oscar6/6F682A90B86D8F9F1/wkb/UniXcoder-base")
    model = AutoModel.from_pretrained("/media/oscar6/6F682A90B86D8F9F1/wkb/UniXcoder-base").to("cuda")
    encoder = EnhancedEncoder(tokenizer, model, "cuda")
    logger = ResultLogger("training_results.json")
    
    optimizer = AdamW([
        {'params': encoder.model.parameters(),'lr':2e-5},
        {'params': encoder.projection.parameters(),'lr': 1e-4},
        {'params': [encoder.temperature],'lr':0.001}
    ],weight_decay=0.1)
    
    best_metrics = {"f1": 0}
    for epoch in range(5):
        print(f"\n==== Epoch {epoch+1} ====")
        train_loss = train_epoch(encoder, train_loader, optimizer, "cuda")

        acc, p, r, f1, best_thresh,TP, TN, FP, FN, detailed = evaluate(encoder, test_loader, "cuda")
        print(f"Train Loss: {train_loss:.4f} | Test Acc: {acc:.4f}, F1: {f1:.4f}, Best Thresh: {best_thresh:.2f}")
        print(f"Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        
        logger.add(epoch+1, {
            "train_loss": float(train_loss),
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "threshold": float(best_thresh),
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        },detailed)
        
        if f1 > best_metrics["f1"]:
            best_metrics.update({"acc": acc, "p": p, "r": r, "f1": f1})
            torch.save({
                'model': encoder.model.state_dict(),
                'projection': encoder.projection.state_dict(),
                'fusion': encoder.fusion_gate.state_dict(),
                'tokenizer': tokenizer
            }, "best_model.pth")
            print("💾 最佳模型已保存")
    
    logger.save()
    print("\n==== 最佳模型表现 ====")
    print(f"Accuracy: {best_metrics['acc']:.4f}")
    print(f"Precision: {best_metrics['p']:.4f}") 
    print(f"Recall: {best_metrics['r']:.4f}")
    print(f"F1: {best_metrics['f1']:.4f}")
