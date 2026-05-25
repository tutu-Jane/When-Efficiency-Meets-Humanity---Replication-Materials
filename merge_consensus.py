import pandas as pd

# 路径换成你自己的实际位置
map_path = r"C:\Users\lenovo\Desktop\phd\AI-tertiary education\AICE\数据\Topic_Theory_Mapping.xlsx"
exp_path = r"C:\Users\lenovo\Desktop\phd\AI-tertiary education\AICE\数据\Expert_Mapping_with_Bucket6_multi.xlsx"
out_path = r"C:\Users\lenovo\Desktop\phd\AI-tertiary education\AICE\数据\Topic_Theory_Mapping_withConsensus.xlsx"

# 读取文件
df_map = pd.read_excel(map_path)
df_exp = pd.read_excel(exp_path)

# 自动识别“topic”列
topic_col_map = [c for c in df_map.columns if "topic" in c.lower()][0]
topic_col_exp = [c for c in df_exp.columns if "topic" in c.lower()][0]

# 只取专家表中的 Consensus_Buckets6 列
df_exp_sub = df_exp[[topic_col_exp, "Consensus_Buckets6"]].rename(columns={topic_col_exp: topic_col_map})

# 合并
df_merged = pd.merge(df_map, df_exp_sub, on=topic_col_map, how="left")

# 保存结果
df_merged.to_excel(out_path, index=False)
print(f"✅ 合并完成：{out_path}")
