"""
MA_insert_att48.py — Memetic Algorithm (模因算法) 求解 TSP/ATT48
Local Search 算子: Insert (将一个城市移动到另一个位置)

MA = GA + Local Search
- GA 负责全局探索: 种群、选择、交叉、变异
- LS 负责局部精化: 对每个后代用 Insert 进行局部搜索
- Lamarckian 进化: LS 改善后的解写回染色体

Insert LS 策略: First-Improvement
  随机采样邻域，接受第一个改进的 Insert 移动，迭代至无法改进或达到上限。
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


# 预计算距离矩阵 (全局变量)
_dist_matrix = None

def build_dist_matrix(cities):
    """预计算 n×n 距离矩阵，避免重复计算"""
    global _dist_matrix
    n = len(cities)
    _dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = calc_distance(cities[i], cities[j])
            _dist_matrix[i][j] = d
            _dist_matrix[j][i] = d


def d(a, b):
    """O(1) 距离查询"""
    return _dist_matrix[a][b]


def calc_total_distance(route, cities=None):
    dist = 0
    n = len(route)
    for i in range(n):
        dist += d(route[i], route[(i + 1) % n])
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
# 3. Local Search: Insert (First-Improvement)
# ============================================================
def calc_insert_delta(route, cities, src, dst):
    """
    O(1) 增量计算 Insert 的距离变化。
    从位置 src 取出城市，插入到位置 dst。
    改变 3 条边。

    若 src < dst:  删除 (src-1,src), (src,src+1), (dst,dst+1)
                    添加 (src-1,src+1), (dst,src), (src,dst+1)
    若 src > dst:  删除 (src-1,src), (src,src+1), (dst-1,dst)
                    添加 (src-1,src+1), (dst-1,src), (src,dst)
    """
    n = len(route)
    if src == dst:
        return 0

    X = route[src]                          # 被移动的城市
    P = route[(src - 1) % n]                # src 的前驱
    Q = route[(src + 1) % n]                # src 的后继

    if src < dst:
        R = route[dst]                      # 插入位置 dst
        S = route[(dst + 1) % n]            # dst 的后继
        old_edges = d(P, X) + d(X, Q) + d(R, S)
        new_edges = d(P, Q) + d(R, X) + d(X, S)
    else:
        R = route[(dst - 1) % n]            # dst 的前驱
        S = route[dst]                      # 插入位置 dst
        old_edges = d(P, X) + d(X, Q) + d(R, S)
        new_edges = d(P, Q) + d(R, X) + d(X, S)

    return new_edges - old_edges


def local_search_insert(route, cities, max_iter=200):
    """
    Insert First-Improvement 局部搜索 (O(1) 增量计算 + 纯随机采样)。
    随机采样 (src, dst) 对，接受第一个改进，直至无法改进或达到上限。
    """
    n = len(cities)
    current = route[:]
    current_dist = calc_total_distance(current)

    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        # 每轮随机采样 n*2 个候选 (src, dst)
        for _ in range(n * 5):
            src = random.randint(0, n - 1)
            dst = random.randint(0, n - 2)
            if dst >= src:
                dst += 1  # 确保 src != dst，均匀分布
            delta = calc_insert_delta(current, cities, src, dst)
            if delta < 0:
                city = current.pop(src)
                current.insert(dst, city)
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

    print(f"[MA+Insert] Pop={pop_size}, Gen={generations}, Pc={pc}, Pm={pm}")
    print(f"[MA+Insert] 初始最优 = {best_dist}")

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

            # ★ MA 核心: Insert Local Search
            c1, _ = local_search_insert(c1, cities)
            c2, _ = local_search_insert(c2, cities)

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
    print(f"[MA+Insert] 最终最优 = {best_dist}, 耗时 = {elapsed:.2f}s")
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
    ax1.set_title(f"MA + Insert — Best Route\nDistance = {best_dist}  (Optimal = {optimal})")
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
    ax2.set_title("MA + Insert — Convergence Curve")
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

    report = f"""# MA + Insert — TSP/ATT48 实验报告

## 算法配置

| 参数 | 值 |
|------|----|
| 问题 | TSP / ATT48 |
| 城市数 | 48 |
| 算法 | **Memetic Algorithm (模因算法) = GA + Insert Local Search** |
| 种群大小 | {params['pop_size']} |
| 进化代数 | {params['generations']} |
| 编码方式 | 排列编码 (Permutation) |
| 选择策略 | 锦标赛选择 (k={params['tournament_k']}) |
| 交叉算子 | OX (Order Crossover), Pc={params['pc']} |
| 变异算子 | Inversion (逆转变异), Pm={params['pm']} |
| 精英保留 | {params['elite_size']} |
| **Local Search** | **Insert First-Improvement (随机采样至局部最优)** |
| LS 策略 | Lamarckian (改善后的基因写回染色体) |
| LS 应用范围 | 所有后代 (offspring) |
| 种群更新 | μ+λ 选择 |
| 随机种子 | {params['seed']} |
| TSPLIB 理论最优 | **{optimal}** |

## 邻域算子说明

**Insert First-Improvement Local Search:**

在路径中选取一个城市，将其移动到另一个位置:
- 操作: pop(i) → insert(j)，相当于子路径的旋转
- 改变 3 条边，保持剩余城市的相对顺序
- First-Improvement 策略: 发现第一个改进就接受
- 迭代至无法改进或达到上限 (200)

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

### MA + Insert 的特点

Insert 算子保持了子路径的相对顺序，适合精细调整:
- 一次 Insert 只改变 3 条边，扰动更加"局部化"
- 邻域大小 = 2256，是所有三种算子中最大的
- 适合在已有较好解的基础上进行微调
- 在 VRP、调度等问题中是比 2-opt 更自然的邻域定义

### 三种 MA 变体的 LS 算子对比

| 特性 | MA + 2-opt | MA + Swap | MA + Insert |
|------|-----------|-----------|-------------|
| LS 搜索策略 | Best-Improvement (全扫描) | First-Improvement | First-Improvement |
| 邻域大小 | 1128 | 1128 | 2256 |
| 改变边数 | 2 | 2~4 | 3 |
| 子路径顺序 | 反转 | 打乱 | 保持(旋转) |
| LS 收敛保证 | 保证局部最优 | 近似 | 近似 |
| TSP 专用性 | ★★★ | ★☆☆ | ★★☆ |
"""

    report += f"""
## 输出图片

![MA + Insert 结果](MA_insert_att48.png)
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
        "generations": 300,
        "pc": 0.9,
        "pm": 0.02,
        "elite_size": 2,
        "tournament_k": 3,
        "seed": 42,
    }

    # 预计算距离矩阵 (加速)
    build_dist_matrix(cities)

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

    plot_results(cities, best_route, best_dist, history, "MA_insert_att48.png")
    generate_report(history, best_dist, best_route, elapsed, params, "MA_insert_report.md")

    print(f"\n=== 完成 ===")
    print(f"最优距离: {best_dist}")
    print(f"与最优解差距: {best_dist - OPTIMAL} ({(best_dist - OPTIMAL) / OPTIMAL * 100:.1f}%)")
    print(f"耗时: {elapsed:.2f}s")
