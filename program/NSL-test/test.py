import time
import torch
import numpy as np

# 尝试导入你的主程序逻辑

from NSL_gpt2_sp import greedy_speculative_generate
from utils import load_encoder_hparams_and_params

def run_pure_benchmark():
    model_dir = "models"
    prompt = "Alan Turing theorized that computers would one day become"
    n_tokens = 40  # 每一轮生成的 token 数
    k_values = list(range(1, 31)) # 测试 K 从 1 到 30
    
    results = []

    print("正在加载模型 (124M & 1558M)...")
    encoder, hparams_draft, draft_params = load_encoder_hparams_and_params("124M", model_dir)
    _, hparams_target, target_params = load_encoder_hparams_and_params("1558M", model_dir)
    input_ids = encoder.encode(prompt)

    print(f"\n开始性能测试...")
    print(f"参数: 生成 {n_tokens} tokens, 测试 K 范围: 1-30")
    print("-" * 40)
    print(f"{'K值':<10} | {'耗时 (秒)':<15}")
    print("-" * 40)

    # 预热一次，防止首次运行的延迟干扰数据
    _ = greedy_speculative_generate(list(input_ids), draft_params, target_params, hparams_draft, hparams_target, 5, K=1)

    for k in k_values:
        # 如果有 GPU 则同步，确保计时准确
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # 运行投机采样
        _ = greedy_speculative_generate(
            list(input_ids), 
            draft_params, 
            target_params, 
            hparams_draft, 
            hparams_target, 
            n_tokens, 
            K=k
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        duration = time.time() - start_time
        results.append((k, duration))
        
        # 实时打印数据
        print(f"{k:<10} | {duration:<15.4f}")

    # 保存数据到 CSV 文件，方便后续处理
    with open("benchmark_results.csv", "w", encoding="utf-8") as f:
        f.write("K,Time\n")
        for k, t in results:
            f.write(f"{k},{t}\n")
    
    print("-" * 40)
    # 找出最优 K
    best_k, min_t = min(results, key=lambda x: x[1])
    print(f"测试完成！数据已保存至 benchmark_results.csv")
    print(f"最优参数: K = {best_k}, 最快耗时 = {min_t:.4f}s")

if __name__ == "__main__":
    run_pure_benchmark()