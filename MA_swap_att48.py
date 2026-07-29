"""
MA_swap_att48.py — Memetic Algorithm (模因算法) 求解 TSP/ATT48
Local Search 算子: Swap (交换两个城市)

MA = GA + Local Search
- GA 负责全局探索: 种群、选择、交叉、变异
- LS 负责局部精化: 对每个后代用 Swap 进行局部搜索
- Lamarckian 进化: LS 改善后的解写回染色体

Swap LS 策略: First-Improvement
  随机采样邻域，接受第一个改进的 Swap 移动，迭代至无法改进或达到上限。
"""

import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OPTIMAL = 10628

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
    n = len(population)
    selected = random.sample(range(n), k)
    best_idx = min(selected, key=lambda i: fitness[i])
    return population[best_idx][:]


def ox_crossover(parent1, parent2):
    n = len(parent1)
    i, j = sorted(random.sample(range(n), 2))
    child1 = [-1] * n
    child1[i : j + 1] = parent1[i : j + 1]
    fill_pos = (j + 1) % n
    for city in parent2[j + 1 :] + parent2[: j + 1]:
        if city not in child1:
            child1[fill_pos] = city
            fill_pos = (fill_pos + 1) % n
    child2 = [-1] * n
    child2[i : j + 1] = parent2[i : j + 1]
    fill_pos = (j + 1) % n
    for city in parent1[j + 1 :] + parent1[: j + 1]:
        if city not in child2:
            child2[fill_pos] = city
            fill_pos = (fill_pos + 1) % n
    return child1, child2


def inversion_mutation(route, pm=0.02):
    route = route[:]
    if random.random() < pm:
        n = len(route)
        i, j = sorted(random.sample(range(n), 2))
        route[i : j + 1] = list(reversed(route[i : j + 1]))
    return route


# ============================================================
# 3. Local Search: Swap (First-Improvement)
# ============================================================
def calc_swap_delta(route, cities, i, j):
    """
    O(1) 增量计算 Swap 的距离变化。
    交换位置 i 和 j 的城市，改变 2 条边 (相邻) 或 4 条边 (不相邻)。
    """
    n = len(route)
    if i == j:
        return 0

    # 确保 i < j 简化处理
    if i > j:
        i, j = j, i

    a = route[(i - 1) % n]
    b = route[i]
    c = route[(i + 1) % n]
    d = route[(j - 1) % n]
    e = route[j]
    f = route[(j + 1) % n]

    if j - i == 1:
        # 相邻: 改变 2 条边
        # 原: a-b, b-c(=e), e-f   →   新: a-e, e-b, b-f
        old_edges = (calc_distance(cities[a], cities[b]) +
                     calc_distance(cities[b], cities[e]) +
                     calc_distance(cities[e], cities[f]))
        new_edges = (calc_distance(cities[a], cities[e]) +
                     calc_distance(cities[e], cities[b]) +
                     calc_distance(cities[b], cities[f]))
    elif i == 0 and j == n - 1:
        # 首尾相邻 (环状)
        old_edges = (calc_distance(cities[route[n-2]], cities[route[n-1]]) +
                     calc_distance(cities[route[n-1]], cities[route[0]]) +
                     calc_distance(cities[route[0]], cities[route[1]]))
        new_edges = (calc_distance(cities[route[n-2]], cities[route[0]]) +
                     calc_distance(cities[route[0]], cities[route[n-1]]) +
                     calc_distance(cities[route[n-1]], cities[route[1]]))
    else:
        # 不相邻: 改变 4 条边
        # 原: a-b, b-c, d-e, e-f   →   新: a-e, e-c, d-b, b-f
        old_edges = (calc_distance(cities[a], cities[b]) +
                     calc_distance(cities[b], cities[c]) +
                     calc_distance(cities[d], cities[e]) +
                     calc_distance(cities[e], cities[f]))
        new_edges = (calc_distance(cities[a], cities[e]) +
                     calc_distance(cities[e], cities[c]) +
                     calc_distance(cities[d], cities[b]) +
                     calc_distance(cities[b], cities[f]))
    return new_edges - old_edges


def local_search_swap(route, cities, max_iter=200):
    """
    Swap First-Improvement 局部搜索 (O(1) 增量计算 + 纯随机采样)。
    随机采样 (i, j) 对，接受第一个改进，直至无法改进或达到上限。
    """
    n = len(cities)
    current = route[:]
    current_dist = calc_total_distance(current, cities)

    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        for _ in range(n * 5):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 2)
            if j >= i:
                j += 1
            delta = calc_swap_delta(current, cities, i, j)
            if delta < 0:
                current[i], current[j] = current[j], current[i]
                current_dist += delta
                improved = True
                iteration += 1
                break
        else:
            iteration += 1

    return current, current_dist


# ============================================================
# 4. Memetic Algorithm 主算法
# ============================================================
def memetic_algorithm(cities, pop_size=100, generations=500,
                       pc=0.9, pm=0.02, elite_size=2, tournament_k=3,
                       seed=42, record_interval=10):
    random.seed(seed)
    n = len(cities)

    # 初始化种群
    population = []
    for _ in range(pop_size):
        route = list(range(n))
        random.shuffle(route)
        population.append(route)

    fitness = [calc_total_distance(ind, cities) for ind in population]

    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_route = population[best_idx][:]
    best_dist = fitness[best_idx]

    history = []
    start_time = time.time()

    print(f"[MA+Swap] Pop={pop_size}, Gen={generations}, Pc={pc}, Pm={pm}")
    print(f"[MA+Swap] 初始最优 = {best_dist}")

    for gen in range(generations):
        offspring = []

        while len(offspring) < pop_size - elite_size:
            p1 = tournament_selection(population, fitness, tournament_k)
            p2 = tournament_selection(population, fitness, tournament_k)

            if random.random() < pc:
                c1, c2 = ox_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = inversion_mutation(c1, pm)
            c2 = inversion_mutation(c2, pm)

            # ★ MA 核心: Swap Local Search
            c1, _ = local_search_swap(c1, cities)
            c2, _ = local_search_swap(c2, cities)

            offspring.append(c1)
            offspring.append(c2)

        offspring = offspring[: pop_size - elite_size]

        elite_indices = sorted(range(pop_size), key=lambda i: fitness[i])[:elite_size]
        elites = [population[i][:] for i in elite_indices]

        combined = elites + offspring
        combined_fitness = [calc_total_distance(ind, cities) for ind in combined]

        sorted_indices = sorted(range(len(combined)), key=lambda i: combined_fitness[i])
        population = [combined[i][:] for i in sorted_indices[:pop_size]]
        fitness = [combined_fitness[i] for i in sorted_indices[:pop_size]]

        if fitness[0] < best_dist:
            best_dist = fitness[0]
            best_route = population[0][:]

        if gen % record_interval == 0 or gen == generations - 1:
            avg_fitness = sum(fitness) / len(fitness)
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
    print(f"[MA+Swap] 最终最优 = {best_dist}, 耗时 = {elapsed:.2f}s")
    return best_route, best_dist, history, elapsed


# ============================================================
# 5. 可视化
# ============================================================
def plot_results(cities, best_route, best_dist, history, output_path, optimal=OPTIMAL):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ordered = [cities[i] for i in best_route] + [cities[best_route[0]]]
    xs = [c[0] for c in ordered]
    ys = [c[1] for c in ordered]
    ax1.plot(xs, ys, "b-", linewidth=0.8, alpha=0.7)
    ax1.scatter(xs[:-1], ys[:-1], c="red", s=20, zorder=5)
    ax1.scatter([xs[0]], [ys[0]], c="green", s=80, marker="*", zorder=6, label="Start")
    ax1.set_title(f"MA + Swap — Best Route\nDistance = {best_dist}  (Optimal = {optimal})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.set_aspect("equal")

    gens = [h[0] for h in history]
    bests = [h[1] for h in history]
    avgs = [h[2] for h in history]
    ax2.plot(gens, bests, "b-", linewidth=1.5, label="Best Distance")
    ax2.plot(gens, avgs, "orange", linewidth=0.8, alpha=0.6, label="Avg Distance")
    ax2.axhline(y=optimal, color="green", linestyle="--", linewidth=1, label=f"TSPLIB Optimal = {optimal}")
    ax2.set_title("MA + Swap — Convergence Curve")
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

    report = f"""# MA + Swap — TSP/ATT48 实验报告

## 算法配置

| 参数 | 值 |
|------|----|
| 问题 | TSP / ATT48 |
| 城市数 | 48 |
| 算法 | **Memetic Algorithm (模因算法) = GA + Swap Local Search** |
| 种群大小 | {params['pop_size']} |
| 进化代数 | {params['generations']} |
| 编码方式 | 排列编码 (Permutation) |
| 选择策略 | 锦标赛选择 (k={params['tournament_k']}) |
| 交叉算子 | OX (Order Crossover), Pc={params['pc']} |
| 变异算子 | Inversion (逆转变异), Pm={params['pm']} |
| 精英保留 | {params['elite_size']} |
| **Local Search** | **Swap First-Improvement (随机采样至局部最优)** |
| LS 策略 | Lamarckian (改善后的基因写回染色体) |
| LS 应用范围 | 所有后代 (offspring) |
| 种群更新 | μ+λ 选择 |
| 随机种子 | {params['seed']} |
| TSPLIB 理论最优 | **{optimal}** |

## 邻域算子说明

**Swap First-Improvement Local Search:**

在路径中随机选取两个位置交换城市，接受第一个改进:
- 随机打乱 (i, j) 对的检查顺序
- 发现第一个改进就接受 (First-Improvement)
- 迭代进行，直到无法改进或达到最大迭代次数 (200)
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

### MA + Swap 的特点

Swap 是最通用的排列邻域算子:
- 一次 Swap 改变 2~4 条边，扰动幅度适中
- 邻域大小 = 1128，用 First-Improvement 策略高效搜索
- 适合作为 MA 的"通用型" LS 引擎

### 与 MA + 2-opt 的对比

| 特性 | MA + 2-opt | MA + Swap |
|------|-----------|-----------|
| LS 搜索策略 | Best-Improvement (全扫描) | First-Improvement (随机采样) |
| LS 每次迭代 | 检查全部 1128 个邻域 | 随机采样直到找到改进 |
| LS 收敛保证 | 保证达到局部最优 | 近似局部最优 |
| TSP 专用性 | ★★★ 专用 | ★☆☆ 通用 |
"""

    report += f"""
## 输出图片

![MA + Swap 结果](MA_swap_att48.png)
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

    plot_results(cities, best_route, best_dist, history, "MA_swap_att48.png")
    generate_report(history, best_dist, best_route, elapsed, params, "MA_swap_report.md")

    print(f"\n=== 完成 ===")
    print(f"最优距离: {best_dist}")
    print(f"与最优解差距: {best_dist - OPTIMAL} ({(best_dist - OPTIMAL) / OPTIMAL * 100:.1f}%)")
    print(f"耗时: {elapsed:.2f}s")
