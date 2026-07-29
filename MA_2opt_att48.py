"""
MA_2opt_att48.py — Memetic Algorithm (模因算法) 求解 TSP/ATT48
Local Search 算子: 2-opt (边交换 / 子路径反转)

MA = GA + Local Search
- GA 负责全局探索 (exploration): 种群、选择、交叉、变异
- LS 负责局部精化 (exploitation): 对每个后代用 2-opt 改善到局部最优
- Lamarckian 进化: LS 改善后的解写回染色体

算法流程:
  1. 初始化种群 (随机排列)
  2. 评估适应度
  3. 每代:
     a. 锦标赛选择 → 选父代
     b. OX 交叉 → 产生后代
     c. Inversion 变异 → 引入扰动
     d. ★ 2-opt Local Search → 将后代改善到局部最优 (MA 核心步骤)
     e. μ+λ 选择 → 下一代种群
  4. 输出最优解
"""

import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OPTIMAL = 10628  # TSPLIB 理论最优

# ============================================================
# 1. 数据加载 & 距离计算
# ============================================================
def load_att48(filepath="att48.csv"):
    cities = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                cities.append((float(parts[0]), float(parts[1])))
    return cities


def calc_distance(city_a, city_b):
    dx = city_a[0] - city_b[0]
    dy = city_a[1] - city_b[1]
    return round(math.sqrt(dx * dx + dy * dy))


def calc_total_distance(route, cities):
    dist = 0
    n = len(route)
    for i in range(n):
        dist += calc_distance(cities[route[i]], cities[route[(i + 1) % n]])
    return dist


# ============================================================
# 2. GA 算子
# ============================================================
def tournament_selection(population, fitness, k=3):
    """锦标赛选择: 随机选 k 个个体，返回最优的那个"""
    n = len(population)
    selected = random.sample(range(n), k)
    best_idx = min(selected, key=lambda i: fitness[i])
    return population[best_idx][:]


def ox_crossover(parent1, parent2):
    """Order Crossover (OX): 保持子路径的相对顺序"""
    n = len(parent1)
    # 随机选择交叉段
    i, j = sorted(random.sample(range(n), 2))

    # 子代 1
    child1 = [-1] * n
    child1[i : j + 1] = parent1[i : j + 1]
    # 从 parent2 中补全剩余基因 (保持顺序)
    fill_pos = (j + 1) % n
    for city in parent2[j + 1 :] + parent2[: j + 1]:
        if city not in child1:
            child1[fill_pos] = city
            fill_pos = (fill_pos + 1) % n

    # 子代 2
    child2 = [-1] * n
    child2[i : j + 1] = parent2[i : j + 1]
    fill_pos = (j + 1) % n
    for city in parent1[j + 1 :] + parent1[: j + 1]:
        if city not in child2:
            child2[fill_pos] = city
            fill_pos = (fill_pos + 1) % n

    return child1, child2


def inversion_mutation(route, pm=0.02):
    """Inversion (逆转变异): 以概率 pm 反转一段子路径"""
    route = route[:]
    if random.random() < pm:
        n = len(route)
        i, j = sorted(random.sample(range(n), 2))
        route[i : j + 1] = reversed(route[i : j + 1])
    return route


# ============================================================
# 3. Local Search 算子: 2-opt (Best-Improvement, 迭代至局部最优)
# ============================================================
def local_search_2opt(route, cities, max_iter=50):
    """
    2-opt Best-Improvement 局部搜索。
    每次扫描全部 O(n²) 邻域，选择最优的 2-opt 移动。
    重复直到无法改进 (局部最优) 或达到最大迭代次数。

    返回: 改善后的路径, 改善后的距离
    """
    n = len(cities)
    current = route[:]
    current_dist = calc_total_distance(current, cities)

    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        best_delta = 0
        best_i, best_j = -1, -1

        # 扫描所有 (i, j) 对，找最佳 2-opt 移动
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue  # 跳过整个环的反转 (恒等变换)

                # O(1) 增量计算 delta
                i_prev = (i - 1) % n
                j_next = (j + 1) % n
                a, b = current[i_prev], current[i]
                c, d = current[j], current[j_next]

                old_edges = (calc_distance(cities[a], cities[b]) +
                             calc_distance(cities[c], cities[d]))
                new_edges = (calc_distance(cities[a], cities[c]) +
                             calc_distance(cities[b], cities[d]))
                delta = new_edges - old_edges

                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        # 执行最佳移动
        if best_delta < 0:
            current[best_i : best_j + 1] = list(reversed(current[best_i : best_j + 1]))
            current_dist += best_delta
            improved = True

    return current, current_dist


# ============================================================
# 4. Memetic Algorithm 主算法
# ============================================================
def memetic_algorithm(cities, pop_size=100, generations=500,
                       pc=0.9, pm=0.02, elite_size=2, tournament_k=3,
                       seed=42, record_interval=10):
    """
    Memetic Algorithm = GA + 2-opt Local Search

    参数:
      cities:         城市坐标列表
      pop_size:       种群大小
      generations:    进化代数
      pc:             交叉概率
      pm:             变异概率
      elite_size:     精英保留数
      tournament_k:   锦标赛选择参数 k
      seed:           随机种子
      record_interval: 记录间隔
    """
    random.seed(seed)
    n = len(cities)

    # --- 初始化种群 ---
    population = []
    for _ in range(pop_size):
        route = list(range(n))
        random.shuffle(route)
        population.append(route)

    # 评估初始种群
    fitness = [calc_total_distance(ind, cities) for ind in population]

    # 初始最优
    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_route = population[best_idx][:]
    best_dist = fitness[best_idx]

    history = []
    start_time = time.time()

    print(f"[MA+2-opt] Pop={pop_size}, Gen={generations}, Pc={pc}, Pm={pm}")
    print(f"[MA+2-opt] 初始最优 = {best_dist}")

    # --- 进化主循环 ---
    for gen in range(generations):
        # 生成后代
        offspring = []

        while len(offspring) < pop_size - elite_size:
            # 选择
            p1 = tournament_selection(population, fitness, tournament_k)
            p2 = tournament_selection(population, fitness, tournament_k)

            # 交叉
            if random.random() < pc:
                c1, c2 = ox_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            # 变异
            c1 = inversion_mutation(c1, pm)
            c2 = inversion_mutation(c2, pm)

            # ★ MA 核心: Local Search (Lamarckian)
            c1, _ = local_search_2opt(c1, cities)
            c2, _ = local_search_2opt(c2, cities)

            offspring.append(c1)
            offspring.append(c2)

        # 截断到需要的数量
        offspring = offspring[: pop_size - elite_size]

        # 精英保留
        elite_indices = sorted(range(pop_size), key=lambda i: fitness[i])[:elite_size]
        elites = [population[i][:] for i in elite_indices]

        # 合并: 精英 + 后代
        combined = elites + offspring
        combined_fitness = [calc_total_distance(ind, cities) for ind in combined]

        # μ+λ 选择: 取最优的 pop_size 个
        sorted_indices = sorted(range(len(combined)), key=lambda i: combined_fitness[i])
        population = [combined[i][:] for i in sorted_indices[:pop_size]]
        fitness = [combined_fitness[i] for i in sorted_indices[:pop_size]]

        # 更新全局最优
        if fitness[0] < best_dist:
            best_dist = fitness[0]
            best_route = population[0][:]

        # 记录收敛数据
        if gen % record_interval == 0 or gen == generations - 1:
            avg_fitness = sum(fitness) / len(fitness)
            # 计算种群多样性 (最佳个体与其余个体在解空间中的差异)
            # 用边重叠率衡量: 最佳个体与种群中其他个体共享边的比例
            diversity = 0.0
            if len(population) > 1:
                best_edges = set()
                for i in range(n):
                    a, b = population[0][i], population[0][(i + 1) % n]
                    best_edges.add((min(a, b), max(a, b)))
                overlaps = []
                for ind in population[1:]:
                    shared = 0
                    for i in range(n):
                        a, b = ind[i], ind[(i + 1) % n]
                        if (min(a, b), max(a, b)) in best_edges:
                            shared += 1
                    overlaps.append(shared / n)
                diversity = 1.0 - (sum(overlaps) / len(overlaps))

            history.append((gen, best_dist, avg_fitness, diversity))

            if gen % (record_interval * 5) == 0 or gen == generations - 1:
                print(f"  [Gen {gen:4d}] Best={best_dist}, Avg={avg_fitness:.1f}, "
                      f"Div={diversity:.3f}")

    elapsed = time.time() - start_time
    print(f"[MA+2-opt] 最终最优 = {best_dist}, 耗时 = {elapsed:.2f}s")
    return best_route, best_dist, history, elapsed


# ============================================================
# 5. 可视化
# ============================================================
def plot_results(cities, best_route, best_dist, history, output_path, optimal=OPTIMAL):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 最优路径
    ordered = [cities[i] for i in best_route] + [cities[best_route[0]]]
    xs = [c[0] for c in ordered]
    ys = [c[1] for c in ordered]
    ax1.plot(xs, ys, "b-", linewidth=0.8, alpha=0.7)
    ax1.scatter(xs[:-1], ys[:-1], c="red", s=20, zorder=5)
    ax1.scatter([xs[0]], [ys[0]], c="green", s=80, marker="*", zorder=6, label="Start")
    ax1.set_title(f"MA + 2-opt — Best Route\nDistance = {best_dist}  (Optimal = {optimal})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.set_aspect("equal")

    # 右图: 收敛曲线
    gens = [h[0] for h in history]
    bests = [h[1] for h in history]
    avgs = [h[2] for h in history]
    ax2.plot(gens, bests, "b-", linewidth=1.5, label="Best Distance")
    ax2.plot(gens, avgs, "orange", linewidth=0.8, alpha=0.6, label="Avg Distance")
    ax2.axhline(y=optimal, color="green", linestyle="--", linewidth=1, label=f"TSPLIB Optimal = {optimal}")
    ax2.set_title("MA + 2-opt — Convergence Curve")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Distance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图片] 已保存至 {output_path}")


# ============================================================
# 6. 生成 Markdown 报告
# ============================================================
def generate_report(history, best_dist, best_route, elapsed, params, output_path, optimal=OPTIMAL):
    gap = best_dist - optimal
    gap_pct = (gap / optimal) * 100

    report = f"""# MA + 2-opt — TSP/ATT48 实验报告

## 算法配置

| 参数 | 值 |
|------|----|
| 问题 | TSP / ATT48 |
| 城市数 | 48 |
| 算法 | **Memetic Algorithm (模因算法) = GA + 2-opt Local Search** |
| 种群大小 | {params['pop_size']} |
| 进化代数 | {params['generations']} |
| 编码方式 | 排列编码 (Permutation) |
| 选择策略 | 锦标赛选择 (k={params['tournament_k']}) |
| 交叉算子 | OX (Order Crossover), Pc={params['pc']} |
| 变异算子 | Inversion (逆转变异), Pm={params['pm']} |
| 精英保留 | {params['elite_size']} |
| **Local Search** | **2-opt Best-Improvement (迭代至局部最优)** |
| LS 策略 | Lamarckian (改善后的基因写回染色体) |
| LS 应用范围 | 所有后代 (offspring) |
| 种群更新 | μ+λ 选择 |
| 随机种子 | {params['seed']} |
| TSPLIB 理论最优 | **{optimal}** |

## 邻域算子说明

**2-opt Best-Improvement Local Search:**

MA 的核心是每一代对后代执行 2-opt 局部搜索:
- 扫描全部 O(n²) = 1128 个邻域解
- 选择距离减少最多的那个 2-opt 移动
- 迭代进行，直到无法改进 (达到局部最优)
- Lamarckian 进化: 改善后的排列直接写回染色体

## 运行结果

| 指标 | 值 |
|------|----|
| 最优距离 | {best_dist} |
| 运行耗时 | {elapsed:.2f}s |
| 与理论最优差距 | {gap} ({gap_pct:.1f}%) |

## 收敛过程

| 代数 | 最优值 | 平均值 | 多样性 (1-重叠率) |
|------|--------|--------|-------------------|
"""

    key_points = []
    if history:
        key_points.append(history[0])
        step = max(1, len(history) // 10)
        for i in range(step, len(history) - 1, step):
            key_points.append(history[i])
        if history[-1] not in key_points:
            key_points.append(history[-1])

    for h in key_points:
        report += f"| {h[0]} | {h[1]} | {h[2]:.1f} | {h[3]:.3f} |\n"

    report += f"""
| {history[-1][0]} (最终) | {best_dist} | {history[-1][2]:.1f} | {history[-1][3]:.3f} |

## 最优路径序列

```
"""

    cities_in_order = [str(c + 1) for c in best_route]
    for k in range(0, len(cities_in_order), 12):
        report += " → ".join(cities_in_order[k : k + 12])
        if k + 12 < len(cities_in_order):
            report += " →\n"
    report += f" → {cities_in_order[0]}  (回到起点)\\n"

    report += """
```

## 算法分析

### MA 相比纯 GA (SGA) 的关键改进

MA 在标准 GA 的基础上增加了 **Local Search** 步骤，使每个后代在被放入种群之前
先通过 2-opt 改善到局部最优:

```
SGA:  Select → Crossover → Mutate → [直接进入种群]
MA:   Select → Crossover → Mutate → [2-opt LS] → [进入种群]
                                      ↑
                                  MA 的核心创新
```

### 2-opt Local Search 的作用

1. **消除交叉边**: 2-opt 直接针对 TSP 中最常见的低效模式——路径交叉
2. **快速收敛**: 2-opt 的 Best-Improvement 策略在 O(n²) 内找到当前最好的改进
3. **Lamarckian 进化**: 学到的"技能"(消除交叉)直接编码到基因中传给下一代

### MA vs SGA vs SA 对比

| 特性 | SGA | SA | MA (本算法) |
|------|-----|----|------------|
| 全局探索 | 种群机制 | 温度机制 | 种群机制 |
| 局部搜索 | 无 | 邻域随机采样 | 系统性邻域搜索至局部最优 |
| 收敛速度 | 慢 | 中 | 快 |
| 解的质量 | 低 | 中 | 高 |
"""

    report += f"""
## 输出图片

![MA + 2-opt 结果](MA_2opt_att48.png)
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[报告] 已保存至 {output_path}")


# ============================================================
# 7. 主程序入口
# ============================================================
if __name__ == "__main__":
    cities = load_att48("att48.csv")
    print(f"加载 ATT48 数据集: {len(cities)} 个城市")
    print(f"TSPLIB 理论最优: {OPTIMAL}")

    params = {
        "pop_size": 100,
        "generations": 500,
        "pc": 0.9,
        "pm": 0.02,
        "elite_size": 2,
        "tournament_k": 3,
        "seed": 42,
    }

    best_route, best_dist, history, elapsed = memetic_algorithm(
        cities,
        pop_size=params["pop_size"],
        generations=params["generations"],
        pc=params["pc"],
        pm=params["pm"],
        elite_size=params["elite_size"],
        tournament_k=params["tournament_k"],
        seed=params["seed"],
        record_interval=10,
    )

    plot_results(cities, best_route, best_dist, history, "MA_2opt_att48.png")
    generate_report(history, best_dist, best_route, elapsed, params, "MA_2opt_report.md")

    print(f"\n=== 完成 ===")
    print(f"最优距离: {best_dist}")
    print(f"与最优解差距: {best_dist - OPTIMAL} ({(best_dist - OPTIMAL) / OPTIMAL * 100:.1f}%)")
    print(f"耗时: {elapsed:.2f}s")
