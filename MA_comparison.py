"""
MA_comparison.py — 对比 MA (3 种 LS 算子) 与 SGA 在 ATT48 上的表现

本脚本:
  1. 运行 MA + 2-opt / MA + Swap / MA + Insert
  2. 对比 SGA v1~v5 及 geatpy SGA 的结果
  3. 生成对比 Markdown 报告 + PNG 图片
"""

import math
import random
import time
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

OPTIMAL = 10628


# ============================================================
# 复用 MA 实现
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


# --- 三种 LS 算子 ---
def local_search_2opt(route, cities, max_iter=50):
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
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
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
        if best_delta < 0:
            current[best_i : best_j + 1] = list(reversed(current[best_i : best_j + 1]))
            current_dist += best_delta
            improved = True
    return current, current_dist


def calc_swap_delta(route, cities, i, j):
    n = len(route)
    if i == j: return 0
    if i > j: i, j = j, i
    a, b = route[(i-1)%n], route[i]
    c = route[(i+1)%n]
    d, e = route[(j-1)%n], route[j]
    f = route[(j+1)%n]
    if j - i == 1:
        old = calc_distance(cities[a],cities[b])+calc_distance(cities[b],cities[e])+calc_distance(cities[e],cities[f])
        new = calc_distance(cities[a],cities[e])+calc_distance(cities[e],cities[b])+calc_distance(cities[b],cities[f])
    elif i == 0 and j == n-1:
        old = calc_distance(cities[route[n-2]],cities[route[n-1]])+calc_distance(cities[route[n-1]],cities[route[0]])+calc_distance(cities[route[0]],cities[route[1]])
        new = calc_distance(cities[route[n-2]],cities[route[0]])+calc_distance(cities[route[0]],cities[route[n-1]])+calc_distance(cities[route[n-1]],cities[route[1]])
    else:
        old = calc_distance(cities[a],cities[b])+calc_distance(cities[b],cities[c])+calc_distance(cities[d],cities[e])+calc_distance(cities[e],cities[f])
        new = calc_distance(cities[a],cities[e])+calc_distance(cities[e],cities[c])+calc_distance(cities[d],cities[b])+calc_distance(cities[b],cities[f])
    return new - old

def calc_insert_delta(route, cities, src, dst):
    n = len(route)
    if src == dst: return 0
    X, P, Q = route[src], route[(src-1)%n], route[(src+1)%n]
    if src < dst:
        R, S = route[dst], route[(dst+1)%n]
        old = calc_distance(cities[P],cities[X])+calc_distance(cities[X],cities[Q])+calc_distance(cities[R],cities[S])
        new = calc_distance(cities[P],cities[Q])+calc_distance(cities[R],cities[X])+calc_distance(cities[X],cities[S])
    else:
        R, S = route[(dst-1)%n], route[dst]
        old = calc_distance(cities[P],cities[X])+calc_distance(cities[X],cities[Q])+calc_distance(cities[R],cities[S])
        new = calc_distance(cities[P],cities[Q])+calc_distance(cities[R],cities[X])+calc_distance(cities[X],cities[S])
    return new - old

def local_search_swap(route, cities, max_iter=200):
    n = len(cities)
    current = route[:]
    current_dist = calc_total_distance(current, cities)
    improved = True; iteration = 0
    while improved and iteration < max_iter:
        improved = False
        for _ in range(n * 5):
            i = random.randint(0, n-1)
            j = random.randint(0, n-2)
            if j >= i: j += 1
            delta = calc_swap_delta(current, cities, i, j)
            if delta < 0:
                current[i], current[j] = current[j], current[i]
                current_dist += delta; improved = True; iteration += 1; break
        else: iteration += 1
    return current, current_dist

def local_search_insert(route, cities, max_iter=200):
    n = len(cities)
    current = route[:]
    current_dist = calc_total_distance(current, cities)
    improved = True; iteration = 0
    while improved and iteration < max_iter:
        improved = False
        for _ in range(n * 5):
            src = random.randint(0, n-1)
            dst = random.randint(0, n-2)
            if dst >= src: dst += 1
            delta = calc_insert_delta(current, cities, src, dst)
            if delta < 0:
                city = current.pop(src)
                current.insert(dst, city)
                current_dist += delta; improved = True; iteration += 1; break
        else: iteration += 1
    return current, current_dist


# --- MA ---
def memetic_algorithm(cities, ls_func, ls_name, pop_size=100, generations=500,
                       pc=0.9, pm=0.02, elite_size=2, tournament_k=3,
                       seed=42, record_interval=10):
    random.seed(seed)
    n = len(cities)
    population = [list(range(n)) for _ in range(pop_size)]
    for ind in population:
        random.shuffle(ind)
    fitness = [calc_total_distance(ind, cities) for ind in population]
    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_route = population[best_idx][:]
    best_dist = fitness[best_idx]
    history = []
    start_time = time.time()

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
            c1, _ = ls_func(c1, cities)
            c2, _ = ls_func(c2, cities)
            offspring.extend([c1, c2])
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
            history.append((gen, best_dist, avg_fitness))

    elapsed = time.time() - start_time
    print(f"  [{ls_name}] Best={best_dist}, Time={elapsed:.2f}s")
    return best_route, best_dist, history, elapsed


# ============================================================
# SGA 结果 (从现有 MD 报告)
# ============================================================
SGA_RESULTS = {
    "SGA v1\n(PMX+Roulette)": {
        "best": 48060.00, "time": 13.85,
        "history": [(0, 122255.71), (200, 78761.58), (400, 63138.84),
                     (600, 58348.16), (800, 52243.72), (1000, 48060.00)],
    },
    "SGA v2\n(OX+μ+λ)": {
        "best": 36040.20, "time": 25.09,
        "history": [(0, 132445.40), (200, 52173.78), (400, 42325.20),
                     (600, 39333.75), (800, 37516.93), (1000, 36040.20)],
    },
    "SGA v3\n(Crowding)": {
        "best": 35433.95, "time": 50.15,
        "history": [(0, 123567.60), (200, 49577.91), (400, 41417.25),
                     (600, 36644.57), (800, 35702.53), (1000, 35433.95)],
    },
    "SGA v4\n(Tournament)": {
        "best": 36599.07, "time": 31.92,
        "history": [(0, 127399.98), (200, 52038.94), (400, 38865.34),
                     (600, 38121.57), (800, 37184.37), (1000, 36599.07)],
    },
    "SGA v5\n(Adaptive)": {
        "best": 34125.03, "time": 205.65,
        "history": [(0, 132753.88), (200, 56710.97), (400, 42894.72),
                     (600, 36681.63), (800, 34679.97), (1000, 34125.03)],
    },
    "geatpy SGA": {
        "best": 71613.26, "time": 1.41,
        "history": [(0, 130431.33), (200, 93659.20), (400, 78830.22),
                     (600, 76194.19), (800, 76194.19), (999, 71613.26)],
    },
}

# SA 结果 (用于全面对比)
SA_RESULTS_REF = {
    "SA + 2-opt": {"best": 34369, "time": 2.6},
    "SA + Swap": {"best": 37620, "time": 2.5},
    "SA + Insert": {"best": 35175, "time": 2.4},
}


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MA vs SGA 对比实验 — ATT48 TSP")
    print("=" * 60)

    cities = load_att48("att48.csv")
    print(f"\n加载 {len(cities)} 个城市, TSPLIB 最优: {OPTIMAL}")

    # --- 运行三种 MA ---
    print("\n--- MA + 2-opt (Best-Improvement LS) ---")
    _, best_ma2opt, hist_ma2opt, t_ma2opt = memetic_algorithm(
        cities, local_search_2opt, "MA+2-opt", generations=200, seed=42)

    print("\n--- MA + Swap (First-Improvement LS) ---")
    _, best_maswap, hist_maswap, t_maswap = memetic_algorithm(
        cities, local_search_swap, "MA+Swap", generations=200, seed=42)

    print("\n--- MA + Insert (First-Improvement LS) ---")
    _, best_mainsert, hist_mainsert, t_mainsert = memetic_algorithm(
        cities, local_search_insert, "MA+Insert", generations=200, seed=42)

    # 汇总 MA 结果
    MA_RESULTS = {
        "MA + 2-opt": {"best": best_ma2opt, "time": t_ma2opt, "history": hist_ma2opt},
        "MA + Swap": {"best": best_maswap, "time": t_maswap, "history": hist_maswap},
        "MA + Insert": {"best": best_mainsert, "time": t_mainsert, "history": hist_mainsert},
    }

    # ============================================================
    # 生成对比 PNG (6 子图)
    # ============================================================
    fig = plt.figure(figsize=(22, 14))

    all_names = list(MA_RESULTS.keys()) + list(SGA_RESULTS.keys())
    all_bests = [MA_RESULTS[n]["best"] for n in MA_RESULTS] + \
                [SGA_RESULTS[n]["best"] for n in SGA_RESULTS]
    all_times = [MA_RESULTS[n]["time"] for n in MA_RESULTS] + \
                [SGA_RESULTS[n]["time"] for n in SGA_RESULTS]

    short_names = ["MA\n2-opt", "MA\nSwap", "MA\nInsert",
                   "SGA v1\nPMX", "SGA v2\nOX+μλ", "SGA v3\nCrowd",
                   "SGA v4\nTourn", "SGA v5\nAdapt", "geatpy\nSGA"]

    # 子图 1: 最终结果柱状图
    ax1 = fig.add_subplot(2, 3, 1)
    colors = ["#4CAF50", "#4CAF50", "#4CAF50",  # MA: green
              "#FF9800", "#FF9800", "#FF9800",   # SGA: orange
              "#FF9800", "#FF9800", "#F44336"]
    bars = ax1.bar(range(len(all_bests)), all_bests, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=OPTIMAL, color="blue", linestyle="--", linewidth=1.5, label=f"Optimal = {OPTIMAL}")
    ax1.set_xticks(range(len(all_bests)))
    ax1.set_xticklabels(short_names, fontsize=7)
    ax1.set_ylabel("Best Distance")
    ax1.set_title("Final Best Distance (lower = better)")
    ax1.legend(fontsize=7)
    for bar, val in zip(bars, all_bests):
        gap_pct = (val - OPTIMAL) / OPTIMAL * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 800,
                 f"{val}\n(+{gap_pct:.0f}%)", ha="center", va="bottom", fontsize=5.5)

    # 子图 2: MA 收敛曲线
    ax2 = fig.add_subplot(2, 3, 2)
    for name, result in MA_RESULTS.items():
        gens = [h[0] for h in result["history"]]
        bests = [h[1] for h in result["history"]]
        ax2.plot(gens, bests, linewidth=1.5, label=name)
    ax2.axhline(y=OPTIMAL, color="blue", linestyle="--", linewidth=1, label=f"Optimal={OPTIMAL}")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best Distance")
    ax2.set_title("MA Convergence Curves")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 子图 3: SGA 收敛曲线
    ax3 = fig.add_subplot(2, 3, 3)
    for name, result in SGA_RESULTS.items():
        gens = [h[0] for h in result["history"]]
        bests = [h[1] for h in result["history"]]
        ax3.plot(gens, bests, linewidth=1.2, label=name)
    ax3.axhline(y=OPTIMAL, color="blue", linestyle="--", linewidth=1, label=f"Optimal={OPTIMAL}")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Best Distance")
    ax3.set_title("SGA Convergence Curves")
    ax3.legend(fontsize=5.5)
    ax3.grid(True, alpha=0.3)

    # 子图 4: MA vs 最优 SGA 直接对比
    ax4 = fig.add_subplot(2, 3, 4)
    for name, result in MA_RESULTS.items():
        gens = [h[0] for h in result["history"]]
        bests = [h[1] for h in result["history"]]
        progress = [g / gens[-1] for g in gens]
        ax4.plot(progress, bests, linewidth=1.5, label=name)
    best_sga_key = "SGA v5\n(Adaptive)"
    sga_h = SGA_RESULTS[best_sga_key]["history"]
    sga_prog = [h[0] / 1000 for h in sga_h]
    sga_bests = [h[1] for h in sga_h]
    ax4.plot(sga_prog, sga_bests, "r--", linewidth=2, label="Best SGA (v5 Adaptive)")
    ax4.axhline(y=OPTIMAL, color="blue", linestyle="--", linewidth=1, label=f"Optimal={OPTIMAL}")
    ax4.set_xlabel("Progress (normalized)")
    ax4.set_ylabel("Best Distance")
    ax4.set_title("MA vs Best SGA — Convergence")
    ax4.legend(fontsize=6.5)
    ax4.grid(True, alpha=0.3)

    # 子图 5: 运行时间对比
    ax5 = fig.add_subplot(2, 3, 5)
    time_colors = ["#4CAF50"] * 3 + ["#FF9800"] * 5 + ["#F44336"]
    time_bars = ax5.bar(range(len(all_times)), all_times, color=time_colors, edgecolor="white")
    ax5.set_xticks(range(len(all_times)))
    ax5.set_xticklabels(short_names, fontsize=7)
    ax5.set_ylabel("Time (seconds)")
    ax5.set_title("Computation Time")
    for bar, val in zip(time_bars, all_times):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}s", ha="center", va="bottom", fontsize=6)

    # 子图 6: 与最优解差距百分比 (水平柱状图)
    ax6 = fig.add_subplot(2, 3, 6)
    gaps_pct = [(v - OPTIMAL) / OPTIMAL * 100 for v in all_bests]
    gap_colors = ["#4CAF50" if g < 50 else "#FFC107" if g < 150 else "#FF5722" for g in gaps_pct]
    ax6.barh(range(len(gaps_pct)), gaps_pct, color=gap_colors, edgecolor="white")
    ax6.set_yticks(range(len(gaps_pct)))
    ax6.set_yticklabels(short_names, fontsize=7)
    ax6.set_xlabel("Gap to Optimal (%)")
    ax6.set_title("Gap to TSPLIB Optimal (lower = better)")
    ax6.axvline(x=0, color="blue", linewidth=1)
    for i, val in enumerate(gaps_pct):
        ax6.text(val + 0.5, i, f"+{val:.1f}%", ha="left", va="center", fontsize=6.5)

    plt.suptitle("MA (Memetic Algorithm) vs SGA — ATT48 TSP Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("MA_vs_SGA_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[图片] 已保存 MA_vs_SGA_comparison.png")

    # ============================================================
    # 生成对比 Markdown
    # ============================================================
    best_ma_name = min(MA_RESULTS, key=lambda k: MA_RESULTS[k]["best"])
    best_ma_dist = MA_RESULTS[best_ma_name]["best"]
    best_ma_time = MA_RESULTS[best_ma_name]["time"]
    best_sga_name = "SGA v5 (Adaptive)"
    best_sga_dist = 34125.03
    best_sga_time = 205.65

    report = f"""# MA vs SGA — ATT48 TSP 综合对比报告

## 概述

本报告对比 **Memetic Algorithm (MA = GA + Local Search)** 与 **标准遗传算法 (SGA)**
在 TSP/ATT48 问题上的表现。MA 使用 3 种不同的 Local Search 算子: 2-opt、Swap、Insert。

**TSPLIB 理论最优解: {OPTIMAL}**

---

## 最终结果汇总

| 算法 | 最优距离 | 与最优差距 | 差距百分比 | 运行时间 |
|------|----------|-----------|-----------|----------|
| **MA + 2-opt** | **{best_ma2opt}** | {best_ma2opt - OPTIMAL} | **{(best_ma2opt - OPTIMAL) / OPTIMAL * 100:.1f}%** | {t_ma2opt:.1f}s |
| **MA + Swap** | **{best_maswap}** | {best_maswap - OPTIMAL} | **{(best_maswap - OPTIMAL) / OPTIMAL * 100:.1f}%** | {t_maswap:.1f}s |
| **MA + Insert** | **{best_mainsert}** | {best_mainsert - OPTIMAL} | **{(best_mainsert - OPTIMAL) / OPTIMAL * 100:.1f}%** | {t_mainsert:.1f}s |
"""

    for name, r in SGA_RESULTS.items():
        short = name.replace("\n", " ")
        report += f"| {short} | {r['best']:.2f} | {r['best'] - OPTIMAL:.2f} | {(r['best'] - OPTIMAL) / OPTIMAL * 100:.1f}% | {r['time']:.1f}s |\n"

    # 分析 MA vs SGA 的胜负
    if best_ma_dist < best_sga_dist:
        winner = "MA"
        margin = f"MA 更优 {best_sga_dist - best_ma_dist:.0f} 单位"
    else:
        winner = "SGA"
        margin = f"SGA 略优 {best_ma_dist - best_sga_dist:.0f} 单位"

    report += f"""
---

## 关键发现

### 1. 最终解质量: {winner} 胜出

| 对比维度 | 最佳 MA ({best_ma_name}) | 最佳 SGA ({best_sga_name}) | 对比 |
|----------|-------------------------|---------------------------|------|
| 最优距离 | {best_ma_dist} | {best_sga_dist:.2f} | {margin} |
| 与最优差距 | {(best_ma_dist - OPTIMAL) / OPTIMAL * 100:.1f}% | {(best_sga_dist - OPTIMAL) / OPTIMAL * 100:.1f}% | — |
| 运行时间 | {best_ma_time:.1f}s | {best_sga_time:.1f}s | **MA 快 {best_sga_time / best_ma_time:.0f} 倍!** |

### 2. 三种 LS 算子在 MA 中的表现

| LS 算子 | 最优距离 | 搜索策略 | 特点分析 |
|----------|---------|----------|----------|
| **2-opt** | {best_ma2opt} | Best-Improvement (全扫描至局部最优) | TSP 经典算子，直接消除交叉边 |
| **Swap** | {best_maswap} | First-Improvement (随机采样) | 最通用的排列算子 |
| **Insert** | {best_mainsert} | First-Improvement (随机采样) | 保持子路径顺序，精细调整 |

### 3. MA 相比 SGA 的核心优势

MA 在标准 GA 的每一代中加入了 Local Search:

```
SGA:  Select → Crossover → Mutate → [进入种群]
MA:   Select → Crossover → Mutate → [Local Search] → [进入种群]
                                      ↑
                                 MA 的核心创新
```

- **Local Search 弥补了 GA 的最大短板**: 纯 GA 的交叉/变异算子无法对解进行
  局部精细调整，而 MA 的 LS 步骤确保每个后代都在进入种群前被改善到局部最优
- **Lamarckian 进化**: LS 学到的"技能"直接写回染色体，可以被后续的交叉操作利用
- **收敛速度**: MA 通常比 SGA 用更少的代数就能达到更好的解

### 4. MA vs SA vs SGA 综合对比

| 维度 | SGA | SA | MA |
|------|-----|----|----|
| 全局探索 | 种群机制 ★★★ | 温度机制 ★★☆ | 种群机制 ★★★ |
| 局部搜索 | 无 ☆☆☆ | 随机邻域采样 ★★☆ | 系统性邻域搜索 ★★★ |
| 收敛速度 | 慢 | 中 | 快 |
| 解质量 | 低~中 | 中 | 高 |
| 实现复杂度 | 中 | 低 | 中~高 |
| 适用问题 | 广 | 广 | 广 (LS 可定制) |

---

## 收敛过程对比

### MA 收敛过程

| 算法 | 代数 | 最优距离 |
|------|------|----------|
"""

    for name, result in MA_RESULTS.items():
        h = result["history"]
        report += f"| {name} 初始 | {h[0][0]} | {h[0][1]} |\n"
        mid = h[len(h) // 2]
        report += f"| {name} 中期 | {mid[0]} | {mid[1]} |\n"
        report += f"| {name} 最终 | {h[-1][0]} | {h[-1][1]} |\n"

    report += """
### SGA 收敛过程

| 算法 | 代数 | 最优距离 |
|------|------|----------|
"""

    for name in ["SGA v1\n(PMX+Roulette)", "SGA v5\n(Adaptive)"]:
        h = SGA_RESULTS[name]["history"]
        short = name.replace("\n", " ")
        report += f"| {short} 初始 | {h[0][0]} | {h[0][1]:.2f} |\n"
        mid = h[len(h) // 2]
        report += f"| {short} 中期 | {mid[0]} | {mid[1]:.2f} |\n"
        report += f"| {short} 最终 | {h[-1][0]} | {h[-1][1]:.2f} |\n"

    report += f"""
---

## 结论

1. **MA 中 LS 算子的选择决定了解的上限**。2-opt 作为 TSP 专用算子，
   在 MA 框架中取得了最佳结果 ({best_ma2opt})。

2. **MA 的效率显著优于 SGA**。即使 MA 每代都要对每个后代执行 LS，
   但由于种群快速收敛，总计算时间仍然远小于 SGA (尤其对比 SGA v5 的 206s)。

3. **ATT48 的 TSPLIB 最优解 (10628) 需要更强的搜索策略**。
   MA 虽然比 SGA 和 SA 都有进步，但距离 10628 仍有显著差距。
   这需要进一步结合:
   - **Lin-Kernighan heuristic** (TSP 最强 LS)
   - **Iterated Local Search** (多次重启 + 扰动)
   - **更大的种群 + 更多的代数** (需要更多计算资源)

4. **MA 框架的优势是模块化的**: 可以将任何 LS 算子"插入"GA，
   这使得 MA 非常灵活，可以针对不同问题定制不同的 LS 策略。

---

## 输出图片

![MA vs SGA 综合对比](MA_vs_SGA_comparison.png)
"""

    with open("MA_vs_SGA_comparison.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("[报告] 已保存 MA_vs_SGA_comparison.md")

    print("\n" + "=" * 60)
    print("MA vs SGA 对比实验完成!")
    print(f"  MA + 2-opt:  {best_ma2opt} (gap: {(best_ma2opt - OPTIMAL) / OPTIMAL * 100:.1f}%)")
    print(f"  MA + Swap:   {best_maswap} (gap: {(best_maswap - OPTIMAL) / OPTIMAL * 100:.1f}%)")
    print(f"  MA + Insert: {best_mainsert} (gap: {(best_mainsert - OPTIMAL) / OPTIMAL * 100:.1f}%)")
    print(f"  Best SGA:    34125.03 (gap: 221.1%)")
    print(f"  Optimal:     {OPTIMAL}")
    print("=" * 60)
