#使用传统TF_IDF模型检查bug报告和与源代码文件相关性
import json
import re
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def enhanced_preprocessor(text):
    """增强型文本预处理，保留代码特征"""
    # 处理驼峰命名（保留原始和分割两种形式）
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # 保留特殊符号组合（如verilog中的信号声明）
    text = re.sub(r'([\w])_([\w])', r'\1 \2', text)
    # 合并数字与单位（如10ns -> 10_ns）
    text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1_\2', text)
    # 去除纯数字和单字符
    return ' '.join([word.lower() for word in re.findall(r'\b[\w.]+\b', text) 
                    if len(word) > 1 or word in {'x', 'z'}])  # 保留verilog特殊值

def load_verilog_files(root_dir):
    """载入所有Verilog/SystemVerilog代码文件"""
    code_files = {}
    extensions = (".c", ".cc", ".cpp", ".h", ".hpp", ".py", ".ts", ".js", ".java", ".lex")
    
    for file_path in Path(root_dir).rglob('*'):
        if file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                # 移除块注释但保留行内注释
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                code_files[str(file_path)] = content
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
    return code_files

# 载入数据
bug_report_path = "/media/oscar6/6F682A90B86D8F9F1/wkb/data/Iverilog/bug_report/bug_report_2.json"
code_dir = "/media/oscar6/6F682A90B86D8F9F1/wkb/data/only_code/iverilog"

# 读取bug报告
with open(bug_report_path, "r", encoding="utf-8") as f:
    bug_report_data = json.load(f)
    summary = bug_report_data.get("Summary", "")
    description = bug_report_data.get("Description", "")
    bug_report = f"{summary} {description}".strip()
    print("bug报告:",bug_report)

# 载入所有代码文件
code_files = load_verilog_files(code_dir)
print(f"Loaded {len(code_files)} code files")

# 预处理所有文本
all_texts = [enhanced_preprocessor(bug_report)] + [enhanced_preprocessor(c) for c in code_files.values()]

# 构建TF-IDF模型（针对Verilog优化）
vectorizer = TfidfVectorizer(
    preprocessor=lambda x: x,
    token_pattern=r'(?u)\b\w[\w.]*\b',  # 允许包含点号（如模块引用）
    ngram_range=(1, 3),
    max_features=10000,
    stop_words=None  # 不过滤技术术语
)

tfidf_matrix = vectorizer.fit_transform(all_texts)
# 分离bug报告和代码的向量
bug_vector = tfidf_matrix[0]
print("bug报告向量:",bug_vector)
code_vectors = tfidf_matrix[1:]

# 计算相似度
similarities = cosine_similarity(bug_vector, code_vectors).flatten()

# 生成排序结果
file_list = list(code_files.keys())
sorted_indices = similarities.argsort()[::-1]

# 输出可解释的结果
print(f"\nBug Report: {bug_report[:200]}...\n")
print("Top 10 Matches:")
for i in sorted_indices[:30]:
    score = similarities[i]
    file_path = file_list[i]
    # 计算匹配词权重
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices_feature = bug_vector.toarray().argsort()[0][::-1]
    top_keywords = [feature_names[j] for j in sorted_indices_feature[:5]]
    
    print(f"{Path(file_path).name}")
    print(f"  Similarity: {score:.4f}")
    print(f"  Path: {file_path}")
    print(f"  Keywords: {', '.join(top_keywords)}\n")