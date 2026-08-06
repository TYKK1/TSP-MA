"""
Memetic Algorithm (MA) for TSP — ATT48  [v3: 3-opt 局部搜索]
=================================================================
v3 改进点: 将 v2 的 2-opt 局部搜索替换为 3-opt.

2-opt vs 3-opt:
  2-opt: 删除 2 条边, 1 种重连 → 局部最优 ≈ 33522
  3-opt: 删除 3 条边, 7 种重连 → 搜索邻域更大, 可跳出 2-opt 局部最优

控制变量: 除局部搜索外, 所有参数与 v2 完全一致.

性能优化:
  - 预生成所有三元组 (i,j,k), 每次扫描时 shuffle 避免偏差
  - 7 种 delta 用纯查表 O(1) 计算
  - max_ls_improvements=5 (3-opt 每次改进比 2-opt 大得多)

Author: TYKK1
Date:   2026-08-06
"""

import csv
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================================
# 0. 全局配置 (框架参数与 v2 一致)
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
    "max_generations": 500,
    # --- 遗传算子 (与 v2 一致) ---
    "crossover": "ERX (Edge Recombination Crossover)",
    "Pc": 0.9,
    "mutation": "Inversion (逆转变异)",
    "Pm": 0.05,
    # --- 选择 (与 v2 一致) ---
    "selection": "Tournament (k=3)",
    "tournament_k": 3,
    "elitism": 2,
    # --- 学习 (局部搜索) — v3: 3-opt ---
    "local_search": "3-opt (first-improvement, 7 patterns, stochastic sampling)",
    "Pls": 0.3,
    "max_ls_improvements": 5,        # 3-opt 每次改进力度大, 5 次足够
    "ls_sample_ratio": 0.2,          # 每次扫描抽查 20% 三元组 (提速 5x)
    "lamarckian": True,
    # --- 输出 ---
    "output_dir": ".",
}

# ============================================================================
# 1. 数据加载
# ============================================================================

def load_att48(csv_path):
    coords = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            coords.append((float(row[0]), float(row[1])))
    if len(coords) != CONFIG["num_cities"]:
        raise ValueError(f"期望 {CONFIG['num_cities']} 个城市, 实际读取 {len(coords)} 个")
    return coords


def compute_dist_matrix(coords):
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            xj, yj = coords[j]
            dist[i][j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
    return dist


# ============================================================================
# 2. 适应度评估
# ============================================================================

def tour_distance(tour, dist):
    total = 0.0
    n = len(tour)
    for i in range(n - 1):
        total += dist[tour[i]][tour[i + 1]]
    total += dist[tour[-1]][tour[0]]
    return total


def evaluate_population(population, dist):
    return [tour_distance(ind, dist) for ind in population]


# ============================================================================
# 3. 初始化
# ============================================================================

def init_population(pop_size, num_cities):
    pop = []
    for _ in range(pop_size):
        tour = list(range(num_cities))
        random.shuffle(tour)
        pop.append(tour)
    return pop


# ============================================================================
# 4. 选择 — 锦标赛
# ============================================================================

def tournament_select(population, fitness, k, num_parents):
    n = len(population)
    pool = []
    for _ in range(num_parents):
        cand = random.sample(range(n), k)
        best = min(cand, key=lambda i: fitness[i])
        pool.append(population[best][:])
    return pool


# ============================================================================
# 5. 交叉 — ERX (与 v2 一致)
# ============================================================================

def build_edge_map(p1, p2):
    n = len(p1)
    em = defaultdict(set)

    def add(tour):
        for i in range(n):
            u = tour[i]
            em[u].add(tour[(i - 1) % n])
            em[u].add(tour[(i + 1) % n])

    add(p1)
    add(p2)
    return dict(em)


def erx_crossover(p1, p2):
    n = len(p1)
    em = build_edge_map(p1, p2)
    visited = set()
    child = []
    cur = p1[0] if random.random() < 0.5 else p2[0]

    for _ in range(n):
        child.append(cur)
        visited.add(cur)
        for nb in em.get(cur, set()):
            if nb in em:
                em[nb].discard(cur)
        remaining = [nb for nb in em.get(cur, set()) if nb not in visited]
        if remaining:
            cur = min(remaining, key=lambda nb: len(em.get(nb, set())))
        else:
            unvisited = [c for c in range(n) if c not in visited]
            if unvisited:
                cur = random.choice(unvisited)
            else:
                break
    return child


# ============================================================================
# 6. 变异 — Inversion (与 v2 一致)
# ============================================================================

def inversion_mutation(tour, Pm):
    if random.random() >= Pm:
        return tour
    n = len(tour)
    a, b = sorted(random.sample(range(n), 2))
    tour[a : b + 1] = reversed(tour[a : b + 1])
    return tour


# ============================================================================
# 7. 学习 (Learning) — 3-opt 局部搜索 [v3 核心 — 优化版]
# ============================================================================

def _precompute_triples(n: int) -> list[tuple[int, int, int]]:
    """
    预生成所有三元组 (i, j, k).
    三个切边位置: i, j, k (0 ≤ i < j < k < n)
    约束: j ≥ i+2, k ≥ j+2 (段非空, 边不共享顶点)
    """
    triples = []
    for i in range(n):
        for j in range(i + 2, n):
            for k in range(j + 2, n):
                # 排除第三边与第一边重合的情况 (k=n-1 且 i=0)
                if k == n - 1 and i == 0:
                    continue
                triples.append((i, j, k))
    return triples


# 全局缓存: 三元组列表 (模块加载时生成)
_ALL_TRIPLES = None


def _get_triples(n: int) -> list[tuple[int, int, int]]:
    global _ALL_TRIPLES
    if _ALL_TRIPLES is None:
        _ALL_TRIPLES = _precompute_triples(n)
    return _ALL_TRIPLES


def three_opt_improve(tour, dist, max_improvements, sample_ratio=0.2):
    """
    3-opt first-improvement (随机采样加速版).

    每次扫描随机抽取 sample_ratio 比例的三元组, 找到改进就应用.
    若抽查样本中无改进, 以高概率认为个体已局部最优.

    返回: (改进后的 tour, 实际改进次数)
    """
    n = len(tour)
    t = tour[:]
    all_triples = _get_triples(n)
    d = dist
    improvements = 0

    # 每次抽查的三元组数量
    sample_size = max(1000, int(len(all_triples) * sample_ratio))

    REV_FLAGS = {
        1: (True, False, False),  2: (False, True, False),
        3: (False, False, True),  4: (True, True, False),
        5: (True, False, True),   6: (False, True, True),
        7: (True, True, True),
    }

    for _ in range(max_improvements):
        improved = False
        # 随机采样三元组
        if sample_size < len(all_triples):
            sample = random.sample(all_triples, sample_size)
        else:
            sample = all_triples
            random.shuffle(sample)

        for i, j, k in sample:
            i1 = i + 1
            j1 = j + 1
            k1 = (k + 1) % n

            ti, ti1 = t[i], t[i1]
            tj, tj1 = t[j], t[j1]
            tk, tk1 = t[k], t[k1]

            old = d[ti][ti1] + d[tj][tj1] + d[tk][tk1]

            best_delta = 0.0
            best_pat = -1

            # 模式1-3: 等价 2-opt
            delta = d[ti][tj] + d[ti1][tj1] + d[tk][tk1] - old
            if delta < best_delta:
                best_delta = delta; best_pat = 1

            delta = d[ti][ti1] + d[tj][tk] + d[tj1][tk1] - old
            if delta < best_delta:
                best_delta = delta; best_pat = 2

            delta = d[tk1][ti1] + d[tj][tj1] + d[tk][ti] - old
            if delta < best_delta:
                best_delta = delta; best_pat = 3

            # 模式4-7: 真 3-opt
            delta = d[ti][tj] + d[ti1][tk] + d[tj1][tk1] - old
            if delta < best_delta:
                best_delta = delta; best_pat = 4

            delta = d[tk1][tj] + d[ti1][tj1] + d[tk][ti] - old
            if delta < best_delta:
                best_delta = delta; best_pat = 5

            delta = d[tk1][ti1] + d[tj][tk] + d[tj1][ti] - old
            if delta < best_delta:
                best_delta = delta; best_pat = 6

            delta = d[tk1][tj] + d[ti1][tk] + d[tj1][ti] - old
            if delta < best_delta:
                best_delta = delta
                best_pat = 7

            if best_pat >= 0:
                A = t[i1 : j + 1]
                B = t[j1 : k + 1]
                C = t[: i + 1] if k1 == 0 else t[k1:] + t[: i + 1]

                ra, rb, rc = REV_FLAGS[best_pat]
                if ra:
                    A.reverse()
                if rb:
                    B.reverse()
                if rc:
                    C.reverse()

                t = C + A + B
                improvements += 1
                improved = True
                break  # first-improvement

        if not improved:
            break  # 采样中无改进 → 假定局部最优

    return t, improvements


# ============================================================================
# 8. 新世代生成 (与 v2 一致)
# ============================================================================

def generate_next_population(offspring, offspring_fitness, elitism, pop_size):
    combined = list(zip(offspring, offspring_fitness))
    combined.sort(key=lambda x: x[1])
    new_pop = []
    for i in range(min(elitism, len(combined))):
        new_pop.append(combined[i][0][:])
    for ind, _ in combined:
        if len(new_pop) >= pop_size:
            break
        if ind not in new_pop:
            new_pop.append(ind[:])
    nc = len(combined[0][0]) if combined else CONFIG["num_cities"]
    while len(new_pop) < pop_size:
        r = list(range(nc))
        random.shuffle(r)
        new_pop.append(r)
    return new_pop, evaluate_population(new_pop, dist_matrix)


# ============================================================================
# 9. 多样性 & 边继承 (与 v2 一致)
# ============================================================================

def compute_diversity(population):
    if len(population) <= 1:
        return 0.0
    n = len(population[0])
    total = 0.0
    pairs = 0
    max_pairs = 50
    indices = list(range(len(population)))
    sampled = random.sample(indices, min(max_pairs, len(population)))
    for ia in range(len(sampled)):
        for ib in range(ia + 1, len(sampled)):
            a = population[sampled[ia]]
            b = population[sampled[ib]]
            ea = set()
            for i in range(n):
                u, v = a[i], a[(i + 1) % n]
                ea.add((min(u, v), max(u, v)))
            ov = 0
            for i in range(n):
                u, v = b[i], b[(i + 1) % n]
                if (min(u, v), max(u, v)) in ea:
                    ov += 1
            total += ov / n
            pairs += 1
    return 1.0 - (total / pairs) if pairs else 0.0


def count_edges_from_parents(child, p1, p2):
    n = len(child)
    pe = set()
    for tour in [p1, p2]:
        for i in range(n):
            u, v = tour[i], tour[(i + 1) % n]
            pe.add((min(u, v), max(u, v)))
    inh = 0
    for i in range(n):
        u, v = child[i], child[(i + 1) % n]
        if (min(u, v), max(u, v)) in pe:
            inh += 1
    return inh, n


# ============================================================================
# 10. 主循环 — MA
# ============================================================================

def run_ma(dist):
    pop_size = CONFIG["pop_size"]
    max_gen = CONFIG["max_generations"]
    num_cities = CONFIG["num_cities"]
    num_parents = pop_size * 2

    population = init_population(pop_size, num_cities)
    fitness = evaluate_population(population, dist)

    history = {
        "gen": [], "best": [], "avg": [], "diversity": [], "edge_inheritance": [],
    }
    best_tour = None
    best_distance = float("inf")
    total_ls_improvements = 0
    stagnation_counter = 0

    # 预生成三元组 (在计时外)
    triples = _get_triples(num_cities)
    print(f"  预生成 {len(triples)} 个三元组用于 3-opt 搜索")
    print("-" * 60)

    start_time = time.perf_counter()

    for gen in range(max_gen + 1):
        current_best = min(fitness)
        current_avg = sum(fitness) / len(fitness)
        current_div = compute_diversity(population)

        if current_best < best_distance:
            best_distance = current_best
            best_idx = fitness.index(current_best)
            best_tour = population[best_idx][:]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        history["gen"].append(gen)
        history["best"].append(current_best)
        history["avg"].append(current_avg)
        history["diversity"].append(current_div)
        history["edge_inheritance"].append(0.0)

        # 进度输出
        if gen % 50 == 0:
            print(f"  Gen {gen:4d}: best={current_best:.0f}, avg={current_avg:.0f}, "
                  f"div={current_div:.3f}, stag={stagnation_counter}")

        if gen >= max_gen:
            break

        # Selection
        mating_pool = tournament_select(
            population, fitness, CONFIG["tournament_k"], num_parents
        )

        # Offspring: ERX + Mutation
        offspring = []
        total_inherited = 0
        total_edges = 0
        random.shuffle(mating_pool)
        for p in range(0, len(mating_pool) - 1, 2):
            pa = mating_pool[p]
            pb = mating_pool[p + 1]
            if random.random() < CONFIG["Pc"]:
                c1 = erx_crossover(pa, pb)
                c2 = erx_crossover(pb, pa)
                inh1, n1 = count_edges_from_parents(c1, pa, pb)
                inh2, n2 = count_edges_from_parents(c2, pa, pb)
                total_inherited += inh1 + inh2
                total_edges += n1 + n2
            else:
                c1, c2 = pa[:], pb[:]
            c1 = inversion_mutation(c1, CONFIG["Pm"])
            c2 = inversion_mutation(c2, CONFIG["Pm"])
            offspring.append(c1)
            offspring.append(c2)

        if total_edges > 0:
            history["edge_inheritance"][-1] = total_inherited / total_edges

        # Learning: 3-opt  [v3 核心]
        for idx in range(len(offspring)):
            if random.random() < CONFIG["Pls"]:
                improved_tour, num_imp = three_opt_improve(
                    offspring[idx], dist, CONFIG["max_ls_improvements"],
                    CONFIG["ls_sample_ratio"],
                )
                total_ls_improvements += num_imp
                if CONFIG["lamarckian"]:
                    offspring[idx] = improved_tour

        # Evaluation
        offspring_fitness = evaluate_population(offspring, dist)

        # New generation
        population, fitness = generate_next_population(
            offspring, offspring_fitness, CONFIG["elitism"], pop_size
        )

    elapsed = time.perf_counter() - start_time

    return {
        "best_tour": best_tour,
        "best_distance": best_distance,
        "history": history,
        "elapsed_sec": elapsed,
        "total_ls_improvements": total_ls_improvements,
        "stagnation_counter": stagnation_counter,
    }


# ============================================================================
# 11. 输出 — Markdown 报告 + 收敛图
# ============================================================================

def generate_report(result, base_name):
    history = result["history"]
    best = result["best_distance"]
    optimum = CONFIG["tsplib_optimum"]
    gap = best - optimum
    gap_pct = (gap / optimum) * 100
    learn_type = "Lamarckian" if CONFIG["lamarckian"] else "Baldwinian"
    avg_inh = sum(history["edge_inheritance"]) / max(len(history["edge_inheritance"]), 1)

    v1_best = 33522.00
    v2_best = 33522.00
    sga_v5_best = 34125.03

    # 计算改进幅度
    improv_vs_v2 = v2_best - best
    improv_pct_vs_v2 = (improv_vs_v2 / v2_best) * 100 if v2_best else 0

    md = f"""# MA v3 (3-opt 局部搜索) — TSP/ATT48 实验报告

## v3 改进说明

v1/v2 使用 **2-opt** 局部搜索, 在 **33522** 处陷入局部最优.
v3 将局部搜索升级为 **3-opt**, 其他参数与 v2 完全一致.

### 为什么 3-opt 能跳出 2-opt 的局部最优?

```
2-opt 邻域 ⊂ 3-opt 邻域

2-opt: 删 2 条边 → 1 种重连 → 局部最优 ≈ 33522
3-opt: 删 3 条边 → 7 种重连 → 搜索空间更大

7 种模式:
  模式1-3:  等价于 2-opt (单段翻转)
  模式4-7:  真 3-opt (两段/三段同时翻转)

  → 2-opt 局部最优处, 真 3-opt 仍可能找到改进!
```

### 3-opt 7 种重连模式

```
删除边: (i,i+1), (j,j+1), (k,k+1)
段: A=[i+1..j], B=[j+1..k], C=[k+1..i]

模式1 (revA):        ───→ 等价 2-opt
模式2 (revB):        ───→ 等价 2-opt
模式3 (revC):        ───→ 等价 2-opt
模式4 (revA+revB):   ───→ 真 3-opt ★
模式5 (revA+revC):   ───→ 真 3-opt ★
模式6 (revB+revC):   ───→ 真 3-opt ★
模式7 (revA+revB+revC): ───→ 真 3-opt ★
```

## 算法配置

| 参数 | 值 |
|------|----|
| 问题 | {CONFIG["problem"]} |
| 城市数 | {CONFIG["num_cities"]} |
| 种群大小 | {CONFIG["pop_size"]} |
| 进化代数 | {CONFIG["max_generations"]} |
| 交叉算子 | {CONFIG["crossover"]} |
| 交叉概率 Pc | {CONFIG["Pc"]} |
| 变异算子 | {CONFIG["mutation"]} |
| 变异概率 Pm | {CONFIG["Pm"]} |
| 选择策略 | {CONFIG["selection"]} |
| 精英保留 | {CONFIG["elitism"]} |
| **局部搜索 (Learning)** | **{CONFIG["local_search"]}** ← v3 改动 |
| 局部搜索概率 Pls | {CONFIG["Pls"]} |
| 最大改进次数/个体 | {CONFIG["max_ls_improvements"]} (v2: 30) |
| 学习模式 | {learn_type} |
| 随机种子 | {CONFIG["random_seed"]} |
| TSPLIB 理论最优 | **{optimum}** |

> 注: max_ls_improvements 从 v2 的 30 降为 5. 原因是 3-opt 每次改进的力度 (同时调整 3 条边) 远大于 2-opt, 5 次 3-opt ≈ 30 次 2-opt 的搜索深度.

## 运行结果

| 指标 | 值 |
|------|----|
| 最优距离 | **{best:.2f}** |
| 与理论最优差距 | {gap:.2f} ({gap_pct:.1f}%) |
| 相比 v2 改进 | **{improv_vs_v2:.0f} ({improv_pct_vs_v2:.1f}%)** |
| 运行耗时 | {result["elapsed_sec"]:.2f}s |
| 3-opt 总改进次数 | {result["total_ls_improvements"]} |
| 最终停滞代数 | {result["stagnation_counter"]} |
| 平均边继承率 | {avg_inh:.1%} |

## v1 / v2 / v3 对比 (控制变量)

| 指标 | v1 (OX+2opt) | v2 (ERX+2opt) | v3 (ERX+3opt) |
|------|:---:|:---:|:---:|
| 交叉算子 | OX | ERX | ERX |
| 局部搜索 | 2-opt | 2-opt | **3-opt** |
| 最优距离 | {v1_best:.0f} | {v2_best:.0f} | **{best:.0f}** |
| 与 TSPLIB 差距 | 215.4% | 215.4% | **{gap_pct:.1f}%** |
| LS 改进次数 | 366,909 | 93,365 | {result["total_ls_improvements"]} |
| 运行耗时 | 43.31s | 32.65s | {result["elapsed_sec"]:.2f}s |

## MA vs SGA 对比

| 算法 | 最优距离 | 耗时 | 与最优差距 |
|------|---------|------|-----------|
| geatpy SGA | 71613.26 | 1.41s | 573.8% |
| my_SGA v5 (自适应) | {sga_v5_best:.0f} | 205.65s | 221.1% |
| MA v1 (OX+2opt) | {v1_best:.0f} | 43.31s | 215.4% |
| MA v2 (ERX+2opt) | {v2_best:.0f} | 32.65s | 215.4% |
| **MA v3 (ERX+3opt)** | **{best:.0f}** | {result["elapsed_sec"]:.2f}s | **{gap_pct:.1f}%** |

## 收敛过程

| 代数 | 最优值 | 平均值 | 多样性 | 边继承率 |
|------|--------|--------|--------|----------|
"""
    report_gens = [g for g in [0, 50, 100, 200, 300, 400, 500] if g <= CONFIG["max_generations"]]
    if CONFIG["max_generations"] not in report_gens:
        report_gens.append(CONFIG["max_generations"])

    for g in report_gens:
        idx = g
        if idx < len(history["gen"]):
            md += (
                f"| {g} | {history['best'][idx]:.2f} "
                f"| {history['avg'][idx]:.2f} "
                f"| {history['diversity'][idx]:.3f} "
                f"| {history['edge_inheritance'][idx]:.1%} |\n"
            )
    fi = -1
    md += (
        f"| {history['gen'][fi]} (最终) "
        f"| {history['best'][fi]:.2f} "
        f"| {history['avg'][fi]:.2f} "
        f"| {history['diversity'][fi]:.3f} "
        f"| {history['edge_inheritance'][fi]:.1%} |\n"
    )

    md += f"""
## 最优路径序列

```
{' → '.join(str(c + 1) for c in result['best_tour'])}
```

## 输出图片

![MA v3 结果]({base_name}.png)
"""
    md_path = Path(CONFIG["output_dir"]) / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] 报告已保存: {md_path}")

    # --- 收敛曲线图 (2x3) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    gens = history["gen"]

    ax = axes[0][0]
    ax.plot(gens, history["best"], "b-", lw=1.5, label="Best")
    ax.plot(gens, history["avg"], "orange", lw=1, alpha=0.7, label="Avg")
    ax.axhline(y=optimum, color="green", linestyle="--", lw=1, label=f"Opt ({optimum})")
    ax.set_xlabel("Gen"); ax.set_ylabel("Distance")
    ax.set_title("Convergence"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(gens, history["diversity"], "g-", lw=1.5)
    ax.set_xlabel("Gen"); ax.set_ylabel("Diversity")
    ax.set_title("Diversity"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    ax = axes[0][2]
    ax.plot(gens, [v * 100 for v in history["edge_inheritance"]], "purple", lw=1.5)
    ax.set_xlabel("Gen"); ax.set_ylabel("Inheritance (%)")
    ax.set_title("Edge Inheritance"); ax.set_ylim(0, 105)
    ax.axhline(y=avg_inh * 100, color="red", linestyle="--", lw=1, label=f"Avg: {avg_inh:.1%}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.semilogy(gens, [max(v, 1) for v in history["best"]], "b-", lw=1.5)
    ax.axhline(y=optimum, color="green", linestyle="--", lw=1, label=f"Opt ({optimum})")
    ax.set_xlabel("Gen"); ax.set_ylabel("Distance (log)")
    ax.set_title("Convergence (Log)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][1]
    coords = load_att48(CONFIG["csv_file"])
    tour = result["best_tour"]
    xs = [coords[c][0] for c in tour] + [coords[tour[0]][0]]
    ys = [coords[c][1] for c in tour] + [coords[tour[0]][1]]
    ax.plot(xs, ys, "b-", lw=0.8, alpha=0.7)
    ax.scatter(xs[:-1], ys[:-1], c="red", s=15, zorder=5)
    ax.scatter([xs[0]], [ys[0]], c="green", s=80, zorder=6, label="Start")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"Best Tour ({best:.0f})")
    ax.legend(fontsize=8); ax.set_aspect("equal")

    ax = axes[1][2]
    labels = ["SGA v5", "MA v1\n(OX+2opt)", "MA v2\n(ERX+2opt)", "MA v3\n(ERX+3opt)"]
    vals = [sga_v5_best, v1_best, v2_best, best]
    colors = ["gray", "steelblue", "darkcyan", "darkorange"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.axhline(y=optimum, color="green", linestyle="--", lw=1.5, label=f"Optimum: {optimum}")
    ax.set_ylabel("Best Distance"); ax.set_title("v1 → v2 → v3")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"MA v3 (3-opt) for TSP/ATT48 — Best: {best:.0f}",
        fontsize=14, fontweight="bold",
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
    random.seed(CONFIG["random_seed"])

    print("=" * 60)
    print("MA v3 (3-opt Local Search) for TSP/ATT48")
    print("=" * 60)
    print(f"核心改动: 2-opt → 3-opt (7 patterns, first-improvement)")
    print(f"控制变量: ERX, Pc={CONFIG['Pc']}, Pm={CONFIG['Pm']}, Pls={CONFIG['Pls']}")
    print(f"max_ls_improvements: {CONFIG['max_ls_improvements']} (v2: 30)")
    print("-" * 60)

    coords = load_att48(CONFIG["csv_file"])
    dist_matrix = compute_dist_matrix(coords)
    print(f"已加载 {len(coords)} 个城市, TSPLIB 最优: {CONFIG['tsplib_optimum']}")
    print("-" * 60)

    t0 = time.perf_counter()
    result = run_ma(dist_matrix)
    t1 = time.perf_counter()

    print("-" * 60)
    print(f"运行完成! 总耗时: {t1 - t0:.2f}s")
    print(f"最优距离: {result['best_distance']:.2f}")
    print(f"与最优差距: {result['best_distance'] - CONFIG['tsplib_optimum']:.2f}")
    print(f"3-opt 总改进: {result['total_ls_improvements']}")
    print(f"最终停滞: {result['stagnation_counter']} 代")
    avg_inh = (sum(result['history']['edge_inheritance']) /
               max(len(result['history']['edge_inheritance']), 1))
    print(f"平均边继承率: {avg_inh:.1%}")

    generate_report(result, "MA_TSP_v3_3opt")

    print("=" * 60)
    print("Done!")
    print("=" * 60)
