"""
Memetic Algorithm (MA) for TSP — ATT48
========================================
Based on the MA pseudocode:
  1. Initialization: generate P(0), evaluate fitness
  2. Selection: choose mating pool M(t) from P(t)
  3. Offspring: crossover + mutation → M'(t)
  4. Learning: local search (2-opt) on each offspring  ← KEY MA STEP
  5. Evaluation: compute fitness of M'(t)
  6. Lamarckian update: replace chromosome with improved version
  7. New generation: P(t+1) from P(t) ∪ M'(t)
  8. Return best individual

Author: TYKK1
Date:   2026-08-06
"""

import csv
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================================
# 0. 全局配置
# ============================================================================

CONFIG = {
    # --- 问题 ---
    "problem": "TSP / ATT48",
    "csv_file": "att48.csv",
    "num_cities": 48,
    "tsplib_optimum": 10628,
    "random_seed": 42,
    # --- 种群 ---
    "pop_size": 100,
    "max_generations": 500,          # MA 收敛更快, 500 代足够
    # --- 遗传算子 ---
    "crossover": "OX (Order Crossover)",
    "Pc": 0.9,
    "mutation": "Inversion (逆转变异)",
    "Pm": 0.05,
    # --- 选择 ---
    "selection": "Tournament (k=3)",
    "tournament_k": 3,
    "elitism": 2,                    # 精英保留数
    # --- 学习 (局部搜索) — MA 核心 ---
    "local_search": "2-opt (first-improvement)",
    "Pls": 0.3,                      # 对 30% 子代施加局部搜索
    "max_ls_improvements": 30,       # 每个个体最多改进 30 次
    "lamarckian": True,              # True = 拉马克式; False = 鲍德温式
    # --- 输出 ---
    "output_dir": ".",
}

# ============================================================================
# 1. 数据加载
# ============================================================================

def load_att48(csv_path: str) -> list[tuple[float, float]]:
    """读取 att48.csv, 返回城市坐标列表 (1-indexed: index 0 → city 1)."""
    coords = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            x, y = float(row[0]), float(row[1])
            coords.append((x, y))
    if len(coords) != CONFIG["num_cities"]:
        raise ValueError(f"期望 {CONFIG['num_cities']} 个城市, 实际读取 {len(coords)} 个")
    return coords


def compute_dist_matrix(coords: list[tuple[float, float]]) -> list[list[float]]:
    """计算欧几里得距离矩阵 (四舍五入为整数, 与 TSPLIB 惯例一致)."""
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            xj, yj = coords[j]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            dist[i][j] = round(d)
    return dist


# ============================================================================
# 2. 适应度评估
# ============================================================================

def tour_distance(tour: list[int], dist: list[list[float]]) -> float:
    """计算一条完整回路的距离."""
    total = 0.0
    n = len(tour)
    for i in range(n - 1):
        total += dist[tour[i]][tour[i + 1]]
    total += dist[tour[-1]][tour[0]]  # 回到起点
    return total


def evaluate_population(
    population: list[list[int]], dist: list[list[float]]
) -> list[float]:
    """计算种群中每个个体的适应度 (距离)."""
    return [tour_distance(ind, dist) for ind in population]


# ============================================================================
# 3. 初始化
# ============================================================================

def init_population(pop_size: int, num_cities: int) -> list[list[int]]:
    """随机生成初始种群 (随机排列)."""
    population = []
    for _ in range(pop_size):
        tour = list(range(num_cities))
        random.shuffle(tour)
        population.append(tour)
    return population


# ============================================================================
# 4. 选择 (Selection) — 锦标赛选择
# ============================================================================

def tournament_select(
    population: list[list[int]],
    fitness: list[float],
    k: int,
    num_parents: int,
) -> list[list[int]]:
    """锦标赛选择: 从种群中选出 num_parents 个父代存入 M(t)."""
    n = len(population)
    mating_pool = []
    for _ in range(num_parents):
        # 随机选 k 个, 取最优
        candidates = random.sample(range(n), k)
        best_idx = min(candidates, key=lambda i: fitness[i])
        mating_pool.append(population[best_idx][:])  # 复制
    return mating_pool


# ============================================================================
# 5. 交叉 (Crossover) — Order Crossover (OX)
# ============================================================================

def ox_crossover(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    """OX 交叉: 从两个父代产生两个子代."""
    n = len(parent1)
    # 随机选两个切割点
    a, b = sorted(random.sample(range(n), 2))

    def ox_one(p1: list[int], p2: list[int]) -> list[int]:
        child = [-1] * n
        # 从 p1 复制中间段
        used = set()
        for i in range(a, b + 1):
            child[i] = p1[i]
            used.add(p1[i])
        # 从 p2 按顺序填充剩余位置
        fill_idx = (b + 1) % n
        for city in p2:
            if city not in used:
                child[fill_idx] = city
                fill_idx = (fill_idx + 1) % n
        return child

    child1 = ox_one(parent1, parent2)
    child2 = ox_one(parent2, parent1)
    return child1, child2


# ============================================================================
# 6. 变异 (Mutation) — Inversion Mutation
# ============================================================================

def inversion_mutation(tour: list[int], Pm: float) -> list[int]:
    """逆转变异: 以概率 Pm 随机反转一段子路径."""
    if random.random() >= Pm:
        return tour
    n = len(tour)
    a, b = sorted(random.sample(range(n), 2))
    # 反转 [a, b] 区间
    tour[a : b + 1] = reversed(tour[a : b + 1])
    return tour


# ============================================================================
# 7. 学习 (Learning) — 2-opt 局部搜索 [MA 核心]
# ============================================================================

def two_opt_improve(
    tour: list[int], dist: list[list[float]], max_improvements: int
) -> tuple[list[int], int]:
    """
    2-opt first-improvement 局部搜索:
      遍历所有边对 (i,i+1) 与 (j,j+1),
      若反转 (i+1, j) 段能缩短总距离, 则立即执行 (first-improvement).
      重复直到无改进或达到 max_improvements 次.

    返回: (改进后的 tour, 实际改进次数)
    """
    n = len(tour)
    tour = tour[:]  # 不修改原列表
    improvements = 0

    for _ in range(max_improvements):
        improved = False
        # 随机起点, 避免每次都从同一位置开始
        start_i = random.randrange(n)
        for offset_i in range(n):
            i = (start_i + offset_i) % n
            i_next = (i + 1) % n
            # j 从 i+2 开始 (不相邻的边)
            start_j = (i + 2) % n
            for offset_j in range(2, n - 1):
                j = (start_j + offset_j) % n
                j_next = (j + 1) % n
                # 确保 i 和 j 不重叠
                if j == i or j_next == i:
                    continue
                # 当前边: (tour[i], tour[i+1]) 和 (tour[j], tour[j+1])
                old_cost = dist[tour[i]][tour[i_next]] + dist[tour[j]][tour[j_next]]
                # 新边: (tour[i], tour[j]) 和 (tour[i+1], tour[j+1])
                new_cost = dist[tour[i]][tour[j]] + dist[tour[i_next]][tour[j_next]]
                if new_cost < old_cost:
                    # 反转 (i+1 ... j) 段
                    if i_next <= j:
                        tour[i_next : j + 1] = reversed(tour[i_next : j + 1])
                    else:
                        # 跨数组末尾的情况
                        seg_len = (j - i_next) % n + 1
                        for k in range(seg_len // 2):
                            a_idx = (i_next + k) % n
                            b_idx = (j - k) % n
                            tour[a_idx], tour[b_idx] = tour[b_idx], tour[a_idx]
                    improvements += 1
                    improved = True
                    break  # first-improvement: 立即跳出内层循环
            if improved:
                break  # 重新开始外层循环
        if not improved:
            break  # 局部最优, 停止

    return tour, improvements


# ============================================================================
# 8. 新世代生成 (New Generation)
# ============================================================================

def generate_next_population(
    parents: list[list[int]],
    offspring: list[list[int]],
    offspring_fitness: list[float],
    elitism: int,
    pop_size: int,
) -> tuple[list[list[int]], list[float]]:
    """
    从 P(t) 和 M'(t) 中选出下一世代 P(t+1):
      精英保留 + 从联合池中选最优
    """
    # 联合: (个体, 适应度)
    combined = list(zip(offspring, offspring_fitness))
    # 按适应度升序 (越小越好)
    combined.sort(key=lambda x: x[1])

    new_pop = []
    # 精英保留: 取最优的 elitism 个
    for i in range(min(elitism, len(combined))):
        new_pop.append(combined[i][0][:])

    # 剩余位置从联合池中选 (已经排好序, 直接取)
    for i in range(len(combined)):
        if len(new_pop) >= pop_size:
            break
        # 避免重复个体 (简单去重: 相同 tour 不重复加入)
        ind = combined[i][0]
        if ind not in new_pop:  # 列表比较可行 (tour 是固定排列)
            new_pop.append(ind[:])

    # 如果还不够 (去重后不足), 随机填充
    while len(new_pop) < pop_size:
        new_ind = list(range(len(parents[0])))
        random.shuffle(new_ind)
        new_pop.append(new_ind)

    new_fitness = evaluate_population(new_pop, dist_matrix)
    return new_pop, new_fitness


# ============================================================================
# 9. 多样性计算
# ============================================================================

def compute_diversity(population: list[list[int]]) -> float:
    """
    计算种群多样性: 1 - 平均边重叠率.
    多样性 = 1 表示完全不同; 多样性 → 0 表示趋于一致.
    """
    if len(population) <= 1:
        return 0.0
    n = len(population[0])
    total_overlap = 0.0
    pairs = 0
    # 随机采样 50 对计算 (避免 O(P²) 太慢)
    max_pairs = 50
    indices = list(range(len(population)))
    sampled = random.sample(indices, min(max_pairs, len(population)))
    for idx_a in range(len(sampled)):
        for idx_b in range(idx_a + 1, len(sampled)):
            a = population[sampled[idx_a]]
            b = population[sampled[idx_b]]
            edges_a = set()
            for i in range(n):
                u, v = a[i], a[(i + 1) % n]
                edges_a.add((min(u, v), max(u, v)))
            overlap = 0
            for i in range(n):
                u, v = b[i], b[(i + 1) % n]
                if (min(u, v), max(u, v)) in edges_a:
                    overlap += 1
            total_overlap += overlap / n
            pairs += 1
    if pairs == 0:
        return 0.0
    avg_overlap = total_overlap / pairs
    return 1.0 - avg_overlap


# ============================================================================
# 10. 主循环 — MA
# ============================================================================

def run_ma(dist: list[list[float]]) -> dict:
    """
    执行 MA 主循环, 返回运行记录.
    """
    pop_size = CONFIG["pop_size"]
    max_gen = CONFIG["max_generations"]
    num_cities = CONFIG["num_cities"]
    num_parents = pop_size * 2  # 产生 pop_size * 2 个子代 (每对父代产生 2 个)

    # --- 步骤 1: 初始化 ---
    # P(t), t = 0
    population = init_population(pop_size, num_cities)
    fitness = evaluate_population(population, dist)

    # 记录
    history = {
        "gen": [],
        "best": [],
        "avg": [],
        "diversity": [],
    }
    best_tour = None
    best_distance = float("inf")

    total_ls_improvements = 0  # 统计局部搜索改进次数

    start_time = time.perf_counter()

    for gen in range(max_gen + 1):
        # --- 记录当前代 ---
        current_best = min(fitness)
        current_avg = sum(fitness) / len(fitness)
        current_div = compute_diversity(population)

        if current_best < best_distance:
            best_distance = current_best
            best_idx = fitness.index(current_best)
            best_tour = population[best_idx][:]

        history["gen"].append(gen)
        history["best"].append(current_best)
        history["avg"].append(current_avg)
        history["diversity"].append(current_div)

        if gen >= max_gen:
            break

        # --- 步骤 2: Selection — M(t) ---
        mating_pool = tournament_select(
            population, fitness, CONFIG["tournament_k"], num_parents
        )

        # --- 步骤 3: Offspring — 重组 + 变异 → M'(t) ---
        offspring = []
        random.shuffle(mating_pool)
        for p in range(0, len(mating_pool) - 1, 2):
            p1 = mating_pool[p]
            p2 = mating_pool[p + 1]
            if random.random() < CONFIG["Pc"]:
                c1, c2 = ox_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            c1 = inversion_mutation(c1, CONFIG["Pm"])
            c2 = inversion_mutation(c2, CONFIG["Pm"])
            offspring.append(c1)
            offspring.append(c2)

        # --- 步骤 4: Learning — 局部搜索 (2-opt) [MA 核心] ---
        for idx in range(len(offspring)):
            if random.random() < CONFIG["Pls"]:
                improved_tour, num_imp = two_opt_improve(
                    offspring[idx], dist, CONFIG["max_ls_improvements"]
                )
                total_ls_improvements += num_imp
                # --- 步骤 5: Lamarckian update ---
                if CONFIG["lamarckian"]:
                    offspring[idx] = improved_tour  # 替换染色体
                # (若 Baldwinian: 只改变适应度, 不改变染色体)

        # --- 步骤 6: Evaluation ---
        offspring_fitness = evaluate_population(offspring, dist)

        # --- 步骤 7: New Generation — P(t+1) ---
        population, fitness = generate_next_population(
            population, offspring, offspring_fitness, CONFIG["elitism"], pop_size
        )

    elapsed = time.perf_counter() - start_time

    return {
        "best_tour": best_tour,
        "best_distance": best_distance,
        "history": history,
        "elapsed_sec": elapsed,
        "total_ls_improvements": total_ls_improvements,
    }


# ============================================================================
# 11. 输出 — Markdown 报告 + 收敛图
# ============================================================================

def generate_report(result: dict, dist: list[list[float]], base_name: str):
    """生成 Markdown 报告和收敛曲线图."""
    history = result["history"]
    best = result["best_distance"]
    optimum = CONFIG["tsplib_optimum"]
    gap = best - optimum
    gap_pct = (gap / optimum) * 100

    # --- Markdown 报告 ---
    ls_desc = "2-opt first-improvement"
    if CONFIG["lamarckian"]:
        learn_type = "Lamarckian (拉马克式 — 改进后替换染色体)"
    else:
        learn_type = "Baldwinian (鲍德温式 — 仅改适应度, 不改染色体)"

    md_content = f"""# MA (Memetic Algorithm) — TSP/ATT48 实验报告

## 算法概述

Memetic Algorithm (MA) = Genetic Algorithm + Local Search.
在标准遗传算法 (SGA) 的基础上, 对每个子代施加**局部搜索 (Learning)**, 并将改进后的个体写回种群 (Lamarckian update),
从而显著加速收敛, 获得更优解.

### MA 伪代码对应关系

| 伪代码步骤 | Python 实现 |
|-----------|------------|
| Initialization | `init_population()` — 随机排列 |
| Selection → M(t) | `tournament_select()` — 锦标赛选择 k={CONFIG["tournament_k"]} |
| Offspring → M'(t) | `ox_crossover()` + `inversion_mutation()` |
| **Learning** | **`two_opt_improve()`** — 2-opt 局部搜索 |
| Evaluation | `tour_distance()` |
| Lamarckian update | 将 2-opt 改进后的 tour 写回染色体 |
| New generation P(t+1) | `generate_next_population()` — μ+λ + 精英保留 |

## 算法配置

| 参数 | 值 |
|------|----|
| 问题 | {CONFIG["problem"]} |
| 城市数 | {CONFIG["num_cities"]} |
| 种群大小 | {CONFIG["pop_size"]} |
| 进化代数 | {CONFIG["max_generations"]} |
| 编码方式 | Permutation (排列编码) |
| 交叉算子 | {CONFIG["crossover"]} |
| 交叉概率 Pc | {CONFIG["Pc"]} |
| 变异算子 | {CONFIG["mutation"]} |
| 变异概率 Pm | {CONFIG["Pm"]} |
| 选择策略 | {CONFIG["selection"]} |
| 精英保留 | {CONFIG["elitism"]} |
| **局部搜索 (Learning)** | **{CONFIG["local_search"]}** |
| **局部搜索概率 Pls** | **{CONFIG["Pls"]}** |
| **最大改进次数/个体** | **{CONFIG["max_ls_improvements"]}** |
| **学习模式** | **{learn_type}** |
| 随机种子 | {CONFIG["random_seed"]} |
| TSPLIB 理论最优 | **{optimum}** |

## 运行结果

| 指标 | 值 |
|------|----|
| 最优距离 | **{best:.2f}** |
| 与理论最优差距 | {gap:.2f} ({gap_pct:.1f}%) |
| 运行耗时 | {result["elapsed_sec"]:.2f}s |
| 局部搜索总改进次数 | {result["total_ls_improvements"]} |

## MA vs SGA 对比

| 算法 | 最优距离 | 耗时 | 与最优差距 |
|------|---------|------|-----------|
| geatpy SGA | 71613.26 | 1.41s | 573.8% |
| my_SGA v1 (PMX+轮盘赌) | 48060.00 | 13.85s | 352.2% |
| my_SGA v4 (OX+锦标赛) | 36599.07 | 31.92s | 244.4% |
| my_SGA v5 (自适应变异) | 34125.03 | 205.65s | 221.1% |
| **MA (本实验)** | **{best:.2f}** | **{result["elapsed_sec"]:.2f}s** | **{gap_pct:.1f}%** |

> MA 通过引入局部搜索 (2-opt), 在相同或更少的代数下获得比 SGA 更优的解.
> Lamarckian 学习将局部搜索的改进 "写入" 染色体, 使得优良的局部结构能被遗传给后代.

## 收敛过程

| 代数 | 最优值 | 平均值 | 多样性 |
|------|--------|--------|--------|
"""
    # 每 100 代 (或最后一代) 记录
    report_gens = [g for g in [0, 50, 100, 200, 300, 400, 500] if g <= CONFIG["max_generations"]]
    if CONFIG["max_generations"] not in report_gens:
        report_gens.append(CONFIG["max_generations"])

    for g in report_gens:
        idx = g  # generation == index
        if idx < len(history["gen"]):
            md_content += (
                f"| {g} | {history['best'][idx]:.2f} "
                f"| {history['avg'][idx]:.2f} "
                f"| {history['diversity'][idx]:.3f} |\n"
            )

    # 最终代
    final_idx = -1
    md_content += (
        f"| {history['gen'][final_idx]} (最终) "
        f"| {history['best'][final_idx]:.2f} "
        f"| {history['avg'][final_idx]:.2f} "
        f"| {history['diversity'][final_idx]:.3f} |\n"
    )

    md_content += f"""
## 最优路径序列

```
{' → '.join(str(c + 1) for c in result['best_tour'])}
```

## 输出图片

![MA 结果]({base_name}.png)
"""
    md_path = Path(CONFIG["output_dir"]) / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] 报告已保存: {md_path}")

    # --- 收敛曲线图 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gens = history["gen"]

    # 左上: 最优值 & 平均值
    ax1 = axes[0][0]
    ax1.plot(gens, history["best"], "b-", linewidth=1.5, label="Best Distance")
    ax1.plot(gens, history["avg"], "orange", linewidth=1, alpha=0.7, label="Avg Distance")
    ax1.axhline(y=optimum, color="green", linestyle="--", linewidth=1, label=f"Optimum ({optimum})")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Tour Distance")
    ax1.set_title("Convergence — Best & Average Distance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右上: 多样性
    ax2 = axes[0][1]
    ax2.plot(gens, history["diversity"], "g-", linewidth=1.5)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Diversity (1 - edge overlap)")
    ax2.set_title("Population Diversity")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # 左下: 对数收敛 (log scale)
    ax3 = axes[1][0]
    best_arr = [max(v, 1) for v in history["best"]]
    ax3.semilogy(gens, best_arr, "b-", linewidth=1.5)
    ax3.axhline(y=optimum, color="green", linestyle="--", linewidth=1, label=f"Optimum ({optimum})")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Tour Distance (log scale)")
    ax3.set_title("Convergence (Log Scale)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 右下: 最优路径
    ax4 = axes[1][1]
    coords = load_att48(CONFIG["csv_file"])
    tour = result["best_tour"]
    xs = [coords[c][0] for c in tour] + [coords[tour[0]][0]]
    ys = [coords[c][1] for c in tour] + [coords[tour[0]][1]]
    ax4.plot(xs, ys, "b-", linewidth=0.8, alpha=0.7)
    ax4.scatter(xs, ys, c="red", s=20, zorder=5)
    ax4.scatter([xs[0]], [ys[0]], c="green", s=100, zorder=6, label="Start")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_title(f"Best Tour (Distance: {best:.0f})")
    ax4.legend()
    ax4.set_aspect("equal")

    plt.suptitle(
        f"MA (Memetic Algorithm) for TSP/ATT48 — "
        f"2-opt Local Search, {CONFIG['max_generations']} generations",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    png_path = Path(CONFIG["output_dir"]) / f"{base_name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图片已保存: {png_path}")


# ============================================================================
# 12. main
# ============================================================================

if __name__ == "__main__":
    # 设置随机种子
    random.seed(CONFIG["random_seed"])

    print("=" * 60)
    print("Memetic Algorithm (MA) for TSP/ATT48")
    print("=" * 60)
    print(f"种群大小: {CONFIG['pop_size']}")
    print(f"最大代数: {CONFIG['max_generations']}")
    print(f"局部搜索: {CONFIG['local_search']}")
    print(f"Pls: {CONFIG['Pls']}, 最大改进次数: {CONFIG['max_ls_improvements']}")
    print(f"学习模式: {'Lamarckian' if CONFIG['lamarckian'] else 'Baldwinian'}")
    print("-" * 60)

    # 加载数据
    coords = load_att48(CONFIG["csv_file"])
    dist_matrix = compute_dist_matrix(coords)
    print(f"已加载 {len(coords)} 个城市")
    print(f"TSPLIB 理论最优: {CONFIG['tsplib_optimum']}")
    print("-" * 60)

    # 运行 MA
    t0 = time.perf_counter()
    result = run_ma(dist_matrix)
    t1 = time.perf_counter()

    print(f"\n运行完成! 总耗时: {t1 - t0:.2f}s")
    print(f"最优距离: {result['best_distance']:.2f}")
    print(f"与理论最优差距: {result['best_distance'] - CONFIG['tsplib_optimum']:.2f}")
    print(f"局部搜索总改进次数: {result['total_ls_improvements']}")

    # 生成报告
    base_name = "MA_TSP_v1"
    generate_report(result, dist_matrix, base_name)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
