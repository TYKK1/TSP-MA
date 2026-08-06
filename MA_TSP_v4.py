"""
Memetic Algorithm (MA) for TSP — ATT48  [v4: 毁灭-重生机制]
===============================================================
v4 改进: 当种群陷入长期停滞时, 触发"毁灭-重生":
  1. 保留精英个体 (不丢失已找到的最优解)
  2. 其余个体完全随机化 (跳出当前吸引盆)
  3. 重新开始进化, 直到下一个局部最优
  4. 多次毁灭后, 从中选出全局最优

原理 (Iterated MA):
  传统 MA 在收敛后, 剩余代数全部浪费在空转上 (ERX 继承率 99%).
  v4 检测到停滞 → 主动毁灭 → 重生 → 有机会进入不同的吸引盆.
  多次重生提供了多次"抽奖"机会, 大大增加找到更优解的概率.

基于 v2 (ERX + 2-opt), 因为:
  - v2 是速度最快的版本 (33s/500 代)
  - 毁灭会让大量代数花在重收敛上, 需要快的局部搜索
  - 多次重启对 2-opt 这种快速 LS 更友好

Author: TYKK1
Date:   2026-08-06
"""

import csv
import math
import random
import time
from collections import defaultdict
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
    "max_generations": 500,
    # --- 遗传算子 ---
    "crossover": "ERX (Edge Recombination Crossover)",
    "Pc": 0.9,
    "mutation": "Inversion (逆转变异)",
    "Pm": 0.05,
    # --- 选择 ---
    "selection": "Tournament (k=3)",
    "tournament_k": 3,
    "elitism": 2,
    # --- 学习 (局部搜索) ---
    "local_search": "2-opt (first-improvement)",
    "Pls": 0.3,
    "max_ls_improvements": 30,
    "lamarckian": True,
    # --- 毁灭-重生机制 [v4 核心] ---
    "stagnation_limit": 40,          # 连续停滞 ≥40 代触发毁灭
    "diversity_threshold": 0.0,      # 设为 0 = 不检查多样性, 仅依赖停滞计数
    "elite_keep_on_destroy": 2,      # 毁灭时保留的精英数
    "max_destructions": 5,           # 最多毁灭次数
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
        raise ValueError(f"期望 {CONFIG['num_cities']} 城市, 读 {len(coords)}")
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

def init_random_tour(num_cities):
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour


def init_population(pop_size, num_cities):
    return [init_random_tour(num_cities) for _ in range(pop_size)]


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
# 5. 交叉 — ERX
# ============================================================================

def build_edge_map(p1, p2):
    n = len(p1)
    em = defaultdict(set)
    def add(tour):
        for i in range(n):
            u = tour[i]
            em[u].add(tour[(i - 1) % n])
            em[u].add(tour[(i + 1) % n])
    add(p1); add(p2)
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
            uv = [c for c in range(n) if c not in visited]
            if uv:
                cur = random.choice(uv)
            else:
                break
    return child


# ============================================================================
# 6. 变异 — Inversion
# ============================================================================

def inversion_mutation(tour, Pm):
    if random.random() >= Pm:
        return tour
    n = len(tour)
    a, b = sorted(random.sample(range(n), 2))
    tour[a : b + 1] = reversed(tour[a : b + 1])
    return tour


# ============================================================================
# 7. 学习 — 2-opt (与 v2 一致)
# ============================================================================

def two_opt_improve(tour, dist, max_improvements):
    n = len(tour)
    t = tour[:]
    improvements = 0
    for _ in range(max_improvements):
        improved = False
        start_i = random.randrange(n)
        for offset_i in range(n):
            i = (start_i + offset_i) % n
            i_next = (i + 1) % n
            start_j = (i + 2) % n
            for offset_j in range(2, n - 1):
                j = (start_j + offset_j) % n
                j_next = (j + 1) % n
                if j == i or j_next == i:
                    continue
                old_c = dist[t[i]][t[i_next]] + dist[t[j]][t[j_next]]
                new_c = dist[t[i]][t[j]] + dist[t[i_next]][t[j_next]]
                if new_c < old_c:
                    if i_next <= j:
                        t[i_next : j + 1] = reversed(t[i_next : j + 1])
                    else:
                        seg_len = (j - i_next) % n + 1
                        for k in range(seg_len // 2):
                            a_i = (i_next + k) % n
                            b_i = (j - k) % n
                            t[a_i], t[b_i] = t[b_i], t[a_i]
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return t, improvements


# ============================================================================
# 8. 新世代生成
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
        r = list(range(nc)); random.shuffle(r)
        new_pop.append(r)
    return new_pop, evaluate_population(new_pop, dist_matrix)


# ============================================================================
# 9. 多样性 & 边继承
# ============================================================================

def compute_diversity(population):
    if len(population) <= 1:
        return 0.0
    n = len(population[0])
    total = 0.0; pairs = 0
    max_pairs = 50
    sampled = random.sample(range(len(population)), min(max_pairs, len(population)))
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
            total += ov / n; pairs += 1
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
# 10. 毁灭-重生 [v4 核心]
# ============================================================================

def destroy_and_rebirth(
    population, fitness, elite_keep, pop_size, num_cities,
):
    """
    毁灭当前种群, 保留 top-elite_keep 个体, 其余完全随机化.

    返回: (新种群, 新适应度, 毁灭日志字符串)
    """
    # 按适应度排序
    ranked = sorted(enumerate(fitness), key=lambda x: x[1])

    new_pop = []
    elite_info = []
    for idx in range(min(elite_keep, len(ranked))):
        orig_idx = ranked[idx][0]
        elite = population[orig_idx][:]
        new_pop.append(elite)
        elite_info.append(f"{ranked[idx][1]:.0f}")

    # 其余个体: 完全随机化
    for _ in range(pop_size - len(new_pop)):
        new_pop.append(init_random_tour(num_cities))

    new_fitness = evaluate_population(new_pop, dist_matrix)
    log = f"    保留精英 [{', '.join(elite_info)}], 其余 {pop_size - elite_keep} 个体随机化"
    return new_pop, new_fitness, log


# ============================================================================
# 11. 主循环 — MA + 毁灭-重生
# ============================================================================

def run_ma(dist):
    pop_size = CONFIG["pop_size"]
    max_gen = CONFIG["max_generations"]
    num_cities = CONFIG["num_cities"]
    num_parents = pop_size * 2

    population = init_population(pop_size, num_cities)
    fitness = evaluate_population(population, dist)

    history = {
        "gen": [], "best": [], "avg": [], "diversity": [],
        "edge_inheritance": [], "destroy_markers": [],  # 记录毁灭发生点
    }
    best_tour = None
    best_distance = float("inf")
    total_ls_improvements = 0
    stagnation_counter = 0

    # v4 毁灭统计
    destruction_count = 0
    destruction_gens = []        # 毁灭发生的代数
    post_destruction_bests = []  # 每次毁灭后找到的最优解
    current_cycle_best = float("inf")
    current_cycle_tour = None

    start_time = time.perf_counter()

    for gen in range(max_gen + 1):
        current_best = min(fitness)
        current_avg = sum(fitness) / len(fitness)
        current_div = compute_diversity(population)

        # 跟踪全局最优
        if current_best < best_distance:
            best_distance = current_best
            best_idx = fitness.index(current_best)
            best_tour = population[best_idx][:]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # 跟踪当前周期最优
        if current_best < current_cycle_best:
            current_cycle_best = current_best
            current_cycle_tour = population[fitness.index(current_best)][:]

        history["gen"].append(gen)
        history["best"].append(current_best)
        history["avg"].append(current_avg)
        history["diversity"].append(current_div)
        history["edge_inheritance"].append(0.0)
        history["destroy_markers"].append(False)

        if gen >= max_gen:
            break

        # --- 毁灭-重生检测 [v4 核心] ---
        div_threshold = CONFIG["diversity_threshold"]
        diversity_ok = (div_threshold <= 0.0) or (current_div < div_threshold)

        should_destroy = (
            destruction_count < CONFIG["max_destructions"]
            and stagnation_counter >= CONFIG["stagnation_limit"]
            and diversity_ok
        )

        if should_destroy:
            destruction_count += 1
            destruction_gens.append(gen)
            post_destruction_bests.append(current_cycle_best)

            print(f"\n  *** 毁灭 #{destruction_count} 触发 @ Gen {gen} ***")
            print(f"      停滞: {stagnation_counter} 代, 多样性: {current_div:.3f}")
            print(f"      当前周期最优: {current_cycle_best:.0f}")

            population, fitness, log_msg = destroy_and_rebirth(
                population, fitness,
                CONFIG["elite_keep_on_destroy"],
                pop_size, num_cities,
            )
            print(log_msg)

            history["destroy_markers"][-1] = True

            # 重置停滞计数和周期最优
            stagnation_counter = 0
            current_cycle_best = float("inf")
            current_cycle_tour = None

            # 重新评估 (适应度已更新)
            continue  # 跳到下一代

        # --- 标准 MA 主循环 ---
        # Selection
        mating_pool = tournament_select(
            population, fitness, CONFIG["tournament_k"], num_parents
        )

        # Offspring: ERX + Mutation
        offspring = []
        total_inherited = 0; total_edges = 0
        random.shuffle(mating_pool)
        for p in range(0, len(mating_pool) - 1, 2):
            pa = mating_pool[p]; pb = mating_pool[p + 1]
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
            offspring.append(c1); offspring.append(c2)

        if total_edges > 0:
            history["edge_inheritance"][-1] = total_inherited / total_edges

        # Learning: 2-opt
        for idx in range(len(offspring)):
            if random.random() < CONFIG["Pls"]:
                improved_tour, num_imp = two_opt_improve(
                    offspring[idx], dist, CONFIG["max_ls_improvements"]
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

    # 记录最后一个周期
    if current_cycle_best < float("inf"):
        post_destruction_bests.append(current_cycle_best)

    return {
        "best_tour": best_tour,
        "best_distance": best_distance,
        "history": history,
        "elapsed_sec": elapsed,
        "total_ls_improvements": total_ls_improvements,
        "stagnation_counter": stagnation_counter,
        "destruction_count": destruction_count,
        "destruction_gens": destruction_gens,
        "post_destruction_bests": post_destruction_bests,
    }


# ============================================================================
# 12. 输出 — Markdown 报告 + 收敛图
# ============================================================================

def generate_report(result, base_name):
    history = result["history"]
    best = result["best_distance"]
    optimum = CONFIG["tsplib_optimum"]
    gap = best - optimum
    gap_pct = (gap / optimum) * 100
    learn_type = "Lamarckian" if CONFIG["lamarckian"] else "Baldwinian"
    avg_inh = sum(history["edge_inheritance"]) / max(len(history["edge_inheritance"]), 1)

    v1_best = 33522; v2_best = 33522; v3_best = 33522; sga_v5 = 34125.03

    # 毁灭周期分析
    dc = result["destruction_count"]
    dg = result["destruction_gens"]
    pdb = result["post_destruction_bests"]

    # 构建毁灭周期表格
    destroy_table = ""
    if dc > 0:
        destroy_table = """
## 毁灭-重生周期记录

| 周期 | 毁灭代数 | 毁灭前最优 | 说明 |
|------|---------|-----------|------|
"""
        for idx in range(dc):
            gen_at = dg[idx]
            best_before = pdb[idx] if idx < len(pdb) else "N/A"
            destroy_table += f"| {idx + 1} | {gen_at} | {best_before:.0f} | 停滞触发 → 毁灭重生 |\n"

        # 最终周期
        if len(pdb) > dc:
            destroy_table += f"| 最终 | — | {pdb[-1]:.0f} | 自然结束 (未触发毁灭) |\n"

    md_content = f"""# MA v4 (毁灭-重生) — TSP/ATT48 实验报告

## v4 改进说明

v1-v3 的共同问题: 种群在 ~50 代就收敛到局部最优 33522, 剩余 **90% 的计算资源全部浪费**.
v4 引入**毁灭-重生机制**: 检测到停滞 → 保留精英 → 其余个体随机化 → 重新进化.

### 毁灭-重生原理

```
传统 MA:  收敛 → 空转 450 代 → 输出
v4 MA:   收敛 → [毁灭] → 收敛 → [毁灭] → 收敛 → 输出 min(所有周期)
                ↓              ↓
           随机化 98 个体   随机化 98 个体
           (跳出盆地 #1)    (跳出盆地 #2)
```

### 触发条件 (AND 逻辑)

1. **停滞 ≥ {CONFIG["stagnation_limit"]} 代** — 全局最优无变化
2. **多样性 < {CONFIG["diversity_threshold"]:.0%}** — 种群已趋同, 交叉失效

### 毁灭操作

- **保留**: {CONFIG["elite_keep_on_destroy"]} 个精英个体 (不丢失已找到的最优解)
- **随机化**: 其余 {CONFIG["pop_size"] - CONFIG["elite_keep_on_destroy"]} 个个体完全随机生成
- **上限**: 最多 {CONFIG["max_destructions"]} 次毁灭

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
| 局部搜索 | {CONFIG["local_search"]} |
| Pls / 最大改进数 | {CONFIG["Pls"]} / {CONFIG["max_ls_improvements"]} |
| **停滞触发阈值** | **{CONFIG["stagnation_limit"]} 代** ← v4 新增 |
| **多样性阈值** | **{CONFIG["diversity_threshold"]:.0%}** ← v4 新增 |
| **毁灭保留精英** | **{CONFIG["elite_keep_on_destroy"]}** ← v4 新增 |
| **最大毁灭次数** | **{CONFIG["max_destructions"]}** ← v4 新增 |
| TSPLIB 理论最优 | **{optimum}** |

## 运行结果

| 指标 | 值 |
|------|----|
| 最优距离 | **{best:.2f}** |
| 与理论最优差距 | {gap:.2f} ({gap_pct:.1f}%) |
| 运行耗时 | {result["elapsed_sec"]:.2f}s |
| 毁灭次数 | {dc} |
| 毁灭发生代数 | {dg if dg else 'N/A'} |
| 各周期最优 | {pdb if pdb else 'N/A'} |
| 2-opt 总改进 | {result["total_ls_improvements"]} |
| 平均边继承率 | {avg_inh:.1%} |
{destroy_table}
## v1 → v2 → v3 → v4 对比

| 指标 | v1 (OX+2opt) | v2 (ERX+2opt) | v3 (ERX+3opt) | v4 (ERX+2opt+毁灭) |
|------|:---:|:---:|:---:|:---:|
| 核心策略 | 基础 MA | 优化交叉 | 强化 LS | **多次重启** |
| 最优距离 | {v1_best:.0f} | {v2_best:.0f} | {v3_best:.0f} | **{best:.0f}** |
| 与 TSPLIB 差距 | 215.4% | 215.4% | 215.4% | **{gap_pct:.1f}%** |
| 运行耗时 | 43s | 33s | 122s | {result["elapsed_sec"]:.0f}s |
| 毁灭次数 | 0 | 0 | 0 | **{dc}** |

## 收敛过程

| 代数 | 最优值 | 平均值 | 多样性 | 边继承率 | 事件 |
|------|--------|--------|--------|----------|------|
"""
    report_gens = [g for g in [0, 50, 100, 200, 300, 400, 500] if g <= CONFIG["max_generations"]]
    if CONFIG["max_generations"] not in report_gens:
        report_gens.append(CONFIG["max_generations"])

    for g in report_gens:
        idx = g
        if idx < len(history["gen"]):
            event = ""
            if history["destroy_markers"][idx]:
                event = "🔥 毁灭"
            elif g in dg:
                event = "🔥 毁灭"
            md_content += (
                f"| {g} | {history['best'][idx]:.2f} "
                f"| {history['avg'][idx]:.2f} "
                f"| {history['diversity'][idx]:.3f} "
                f"| {history['edge_inheritance'][idx]:.1%} "
                f"| {event} |\n"
            )

    fi = -1
    md_content += (
        f"| {history['gen'][fi]} (最终) "
        f"| {history['best'][fi]:.2f} "
        f"| {history['avg'][fi]:.2f} "
        f"| {history['diversity'][fi]:.3f} "
        f"| {history['edge_inheritance'][fi]:.1%} "
        f"| |\n"
    )

    md_content += f"""
## 毁灭-重启可视化说明

收敛曲线图中, **红色虚线**标记毁灭发生的位置.
毁灭后种群平均值会大幅飙升 (因为 98% 个体重新随机化),
随后快速收敛到新的局部最优.

## 最优路径序列

```
{' → '.join(str(c + 1) for c in result['best_tour'])}
```

## 输出图片

![MA v4 结果]({base_name}.png)
"""
    md_path = Path(CONFIG["output_dir"]) / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] 报告已保存: {md_path}")

    # --- 收敛曲线图 (2x3) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    gens = history["gen"]

    # (0,0): 收敛曲线 + 毁灭标记
    ax = axes[0][0]
    ax.plot(gens, history["best"], "b-", lw=1.5, label="Best")
    ax.plot(gens, history["avg"], "orange", lw=1, alpha=0.7, label="Avg")
    # 毁灭标记
    for g in dg:
        ax.axvline(x=g, color="red", linestyle="--", lw=1, alpha=0.7)
    if dg:
        ax.axvline(x=dg[0], color="red", linestyle="--", lw=1, alpha=0.7,
                   label=f"Destroy ({dc}x)")
    ax.axhline(y=optimum, color="green", linestyle="--", lw=1, label=f"Opt ({optimum})")
    ax.set_xlabel("Gen"); ax.set_ylabel("Distance")
    ax.set_title("Convergence + Destruction Markers")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (0,1): 多样性 + 毁灭标记
    ax = axes[0][1]
    ax.plot(gens, history["diversity"], "g-", lw=1.5)
    ax.axhline(y=CONFIG["diversity_threshold"], color="red", linestyle=":", lw=1,
               label=f"Threshold ({CONFIG['diversity_threshold']:.0%})")
    for g in dg:
        ax.axvline(x=g, color="red", linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("Gen"); ax.set_ylabel("Diversity")
    ax.set_title("Diversity (re-diversified at each restart)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (0,2): 边继承率
    ax = axes[0][2]
    ax.plot(gens, [v * 100 for v in history["edge_inheritance"]], "purple", lw=1.5)
    ax.set_xlabel("Gen"); ax.set_ylabel("Inheritance (%)")
    ax.set_title("Edge Inheritance"); ax.set_ylim(0, 105)
    ax.axhline(y=avg_inh * 100, color="red", linestyle="--", lw=1,
               label=f"Avg: {avg_inh:.1%}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0): 对数收敛 + 毁灭标记
    ax = axes[1][0]
    ax.semilogy(gens, [max(v, 1) for v in history["best"]], "b-", lw=1.5)
    for g in dg:
        ax.axvline(x=g, color="red", linestyle="--", lw=1, alpha=0.5)
    ax.axhline(y=optimum, color="green", linestyle="--", lw=1, label=f"Opt ({optimum})")
    ax.set_xlabel("Gen"); ax.set_ylabel("Distance (log)")
    ax.set_title("Convergence (Log)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1): 最优路径
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

    # (1,2): 毁灭周期对比
    ax = axes[1][2]
    if pdb:
        cycle_labels = [f"Cycle {i+1}\n({dg[i] if i < len(dg) else 'end'})"
                        for i in range(len(pdb))]
        x_pos = range(len(pdb))
        colors_cycle = ["steelblue", "darkcyan", "darkorange", "crimson", "purple",
                         "saddlebrown"][:len(pdb)]
        bars = ax.bar(cycle_labels, pdb, color=colors_cycle, edgecolor="black")
        ax.axhline(y=optimum, color="green", linestyle="--", lw=1.5,
                   label=f"Optimum: {optimum}")
        ax.axhline(y=v2_best, color="gray", linestyle=":", lw=1,
                   label=f"v2 best: {v2_best:.0f}")
        ax.set_ylabel("Best Distance"); ax.set_title("Best per Destruction Cycle")
        for bar, val in zip(bars, pdb):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    f"{val:.0f}", ha="center", fontsize=8, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No destructions triggered", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Destruction Cycles")

    plt.suptitle(
        f"MA v4 (Destroy-Rebirth) for TSP/ATT48 — Best: {best:.0f}, "
        f"{dc} destructions",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    png_path = Path(CONFIG["output_dir"]) / f"{base_name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图片已保存: {png_path}")


# ============================================================================
# 13. main
# ============================================================================

if __name__ == "__main__":
    random.seed(CONFIG["random_seed"])

    print("=" * 60)
    print("MA v4 (Destroy-Rebirth Mechanism) for TSP/ATT48")
    print("=" * 60)
    print(f"停滞阈值: {CONFIG['stagnation_limit']} 代")
    print(f"多样性阈值: {CONFIG['diversity_threshold']:.0%}")
    print(f"最大毁灭次数: {CONFIG['max_destructions']}")
    print(f"每次毁灭保留精英: {CONFIG['elite_keep_on_destroy']}")
    print(f"基础算法: ERX + 2-opt")
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
    print(f"全局最优距离: {result['best_distance']:.2f}")
    print(f"与 TSPLIB 差距: {result['best_distance'] - CONFIG['tsplib_optimum']:.2f}")
    print(f"毁灭次数: {result['destruction_count']}")
    print(f"毁灭代数: {result['destruction_gens']}")
    print(f"各周期最优: {[f'{x:.0f}' for x in result['post_destruction_bests']]}")
    print(f"2-opt 总改进: {result['total_ls_improvements']}")
    avg_inh = (sum(result['history']['edge_inheritance']) /
               max(len(result['history']['edge_inheritance']), 1))
    print(f"平均边继承率: {avg_inh:.1%}")

    generate_report(result, "MA_TSP_v4_destroy")

    print("=" * 60)
    print("Done!")
    print("=" * 60)
