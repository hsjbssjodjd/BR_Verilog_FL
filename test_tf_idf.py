import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd

def preprocess_bug_report(text):
    """处理包含Verilog代码的bug报告"""
    # 分离自然语言描述和代码部分
    code_blocks = re.findall(r'```(?:verilog)?\s*(.*?)```', text, re.DOTALL)
    text_blocks = re.sub(r'```.*?```', ' ', text)

    # 处理自然语言部分
    text_processed = re.sub(r'\b(?:error|bug|fix)\b', '', text_blocks.lower())
    text_processed = ' '.join(re.findall(r'\b[\w.]+\b', text_processed))

    # 处理代码部分
    code_processed = ' '.join([
        re.sub(r'\b(input|output|wire|reg)\b', '', c)
        for c in code_blocks
    ]).lower()

    return f"{text_processed} {code_processed}"


def preprocess_cpp_code(text):
    """预处理C++源代码"""
    # 保留关键语法结构
    text = re.sub(r'//.*', '', text)  # 保留行内注释中的技术术语
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # 处理特殊符号
    text = re.sub(r'([<>*&])(\w)', r' \1 \2', text)  # 分离操作符
    text = re.sub(r'(\w)([<>*&])', r'\1 \2 ', text)

    # 保留技术术语
    tokens = re.findall(r'\b[\w_]+\b', text.lower())
    return ' '.join([t for t in tokens if len(t) > 1 or t in {'x', 'i'}])


def load_cpp_files(code_dir):
    """加载所有C++文件"""
    code_files = {}
    extensions = (".c", ".cc", ".cpp", ".h", ".hpp")

    for file_path in tqdm(Path(code_dir).rglob('*'), desc="Loading C++ files"):
        if file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code_files[str(file_path)] = preprocess_cpp_code(f.read())
            except Exception as e:
                print(f"\nError reading {file_path}: {str(e)}")
    return code_files


def process_bug_reports(bug_reports_dir, code_files, results):
    """批量处理bug报告"""
    bug_report_files = list(Path(bug_reports_dir).glob("*.json"))

    for bug_file in tqdm(bug_report_files, desc="Processing bug reports"):
        try:
            with open(bug_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                bug_text = f"{data.get('Summary', '')} {data.get('Description', '')}".strip()

            # 预处理bug报告
            processed_bug = preprocess_bug_report(bug_text)

            # 构建TF-IDF矩阵
            corpus = [processed_bug] + list(code_files.values())
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8000,
                token_pattern=r'(?u)\b\w[\w.]+\b'
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # 计算相似度
            similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
            sorted_indices = similarities.argsort()[::-1][:30]

            # 收集结果
            results[bug_file.name] = []
            file_paths = list(code_files.keys())

            for idx in sorted_indices:
                if similarities[idx] < 0.005:  # 过滤低相似度结果
                    continue

                result = {
                    "file": Path(file_paths[idx]).name,
                    "path": file_paths[idx],
                    "score": float(similarities[idx]),
                    "keywords": vectorizer.get_feature_names_out()[
                        tfidf_matrix[0].toarray().argsort()[0][-5:][::-1]
                    ].tolist()
                }
                results[bug_file.name].append(result)

        except Exception as e:
            print(f"\nError processing {bug_file.name}: {str(e)}")


# 配置路径
bug_reports_dir = "/media/oscar6/6F682A90B86D8F9F1/wkb/data/Iverilog/bug_report"
code_dir = "/media/oscar6/6F682A90B86D8F9F1/wkb/data/only_code/iverilog"

# 初始化数据
print("Initializing...")
code_data = load_cpp_files(code_dir)
results = {}

# 执行处理流程
process_bug_reports(bug_reports_dir, code_data, results)

with open('full_results.json', 'w') as f:
    json.dump(results, f, indent=2)


df = pd.DataFrame([
    {"bug": k, "file": x["file"], "score": x["score"]} 
    for k, v in results.items() 
    for x in v
])
df.to_csv("cross_language_matches.csv", index=False)
# 输出结果示例
print("\nProcessing complete. Sample output:")
for bug_name in list(results.keys())[:1]:
    print(f"\nBug Report: {bug_name}")
    for item in results[bug_name][:3]:
        print(f"  {item['file']} ({item['score']:.4f})")
        print(f"    Keywords: {item['keywords']}")
