"""Numerical workbench. Descriptive results and scenario utilities, not forecasts."""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, vstack


def finite_matrix(values, max_size=100_000):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape or matrix.size > max_size:
        raise ValueError("请输入非空二维矩阵，最多 100,000 个元素。")
    if not np.isfinite(matrix).all():
        raise ValueError("矩阵不能包含空值、无穷大或非数值。")
    return matrix


def numeric_profile(frame):
    """Explicitly report missing/invalid values; use complete cases for PCA."""
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("请至少选择一个数值字段。")
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    summary = numeric.describe().T
    summary["missing"] = numeric.isna().sum()
    summary["missing_rate"] = numeric.isna().mean()
    return numeric, summary


def principal_components(frame):
    numeric, summary = numeric_profile(frame)
    clean = numeric.dropna()
    if len(clean) < 3:
        raise ValueError("PCA 至少需要 3 条完整样本；请检查缺失值或选择其他字段。")
    deviations = clean.std(ddof=1)
    active = deviations[deviations > 0].index
    if len(active) < 2:
        raise ValueError("PCA 至少需要 2 个有变化的数值字段。")
    z = (clean[active] - clean[active].mean()) / deviations[active]
    u, singular, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    # Fix SVD's arbitrary sign for reproducible display.
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])
    vt *= signs[:, None]
    u *= signs
    eigenvalues = singular ** 2 / (len(clean) - 1)
    labels = [f"PC{i + 1}" for i in range(len(singular))]
    return {
        "scores": pd.DataFrame(u * singular, index=clean.index, columns=labels),
        "components": pd.DataFrame(vt.T, index=active, columns=labels),
        "loadings": pd.DataFrame(vt.T * np.sqrt(eigenvalues), index=active, columns=labels),
        "variance_ratio": pd.Series(eigenvalues / eigenvalues.sum(), index=labels),
        "standardized": z,
        "summary": summary,
        "dropped_rows": len(frame) - len(clean),
        "constant_columns": list(deviations[deviations == 0].index),
        "rank": int(np.linalg.matrix_rank(z)),
    }


def matrix_diagnostics(values, approximation_rank=1):
    a = finite_matrix(values)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.linalg.matrix_rank(a))
    k = int(approximation_rank)
    if k != approximation_rank or not 1 <= k <= len(s):
        raise ValueError("近似阶数必须在 1 与矩阵最小维数之间。")
    reconstructed = (u[:, :k] * s[:k]) @ vt[:k]
    norm = np.linalg.norm(a)
    symmetric = a.shape[0] == a.shape[1] and np.allclose(a, a.T)
    return {
        "rank": rank,
        "nullity": a.shape[1] - rank,
        "condition": float(s[0] / s[-1]) if rank == min(a.shape) else float("inf"),
        "singular_values": s,
        "eigenvalues": np.linalg.eigvalsh(a) if symmetric else None,
        "reconstructed": reconstructed,
        "relative_error": float(np.linalg.norm(a - reconstructed) / norm) if norm else 0.0,
        "energy": np.cumsum(s ** 2) / np.sum(s ** 2) if norm else np.zeros_like(s),
    }


@dataclass
class AllocationResult:
    selected: pd.DataFrame
    objective: float
    cost: float
    eligible_count: int
    status: str


def optimize_allocation(candidates: pd.DataFrame, budget: float, contact_limit: int,
                        product_capacities: Mapping[str, int]) -> AllocationResult:
    """Binary assignment: ALLOW only, one offer/customer, budget and capacity bounds.

    utility is an explicitly supplied scenario score, never an expected return.
    """
    required = {"customer_id", "product_id", "suitability_level", "utility", "cost"}
    if not required.issubset(candidates.columns):
        raise ValueError("候选表缺少客户、产品、适当性、效用或成本字段。")
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("预算必须是非负有限数。")
    if not np.isfinite(contact_limit) or contact_limit < 0 or int(contact_limit) != contact_limit:
        raise ValueError("触达上限必须是非负整数。")
    if len(candidates) > 2000:
        raise ValueError("单次优化最多支持 2,000 个候选组合。")
    if candidates[["customer_id", "product_id"]].isna().any().any():
        raise ValueError("客户与产品标识不能为空。")
    if candidates.duplicated(["customer_id", "product_id"]).any():
        raise ValueError("候选组合重复，请先去重。")
    frame = candidates.copy()
    for column in ("utility", "cost"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise ValueError("效用与成本必须是有限数值。")
    if (frame.cost < 0).any():
        raise ValueError("成本不能为负。")
    for capacity in product_capacities.values():
        if not np.isfinite(capacity) or capacity < 0 or int(capacity) != capacity:
            raise ValueError("产品容量必须是非负整数。")
    frame = frame.loc[frame.suitability_level.eq("ALLOW")].reset_index(drop=True)
    missing = set(frame.product_id) - set(product_capacities)
    if missing:
        raise ValueError("请为所有可分配产品设置容量。")
    n = len(frame)
    if not n or not contact_limit:
        return AllocationResult(frame.iloc[:0], 0.0, 0.0, n, "optimal")
    rows = [csr_matrix(frame.cost.to_numpy()[None, :]), csr_matrix(np.ones((1, n)))]
    upper = [budget, contact_limit]
    for column, limits in (("customer_id", None), ("product_id", product_capacities)):
        for identity in frame[column].unique():
            rows.append(csr_matrix(frame[column].eq(identity).to_numpy(dtype=float)[None, :]))
            upper.append(1 if limits is None else limits[identity])
    constraints = LinearConstraint(vstack(rows, format="csc"), -np.inf, upper)
    result = milp(-frame.utility.to_numpy(), integrality=np.ones(n),
                  bounds=Bounds(0, 1), constraints=constraints,
                  options={"time_limit": 10.0, "mip_rel_gap": 0.0})
    if not result.success:
        raise ValueError(f"未取得最优解（状态 {result.status}），请缩小候选范围或简化约束。")
    selected = frame.loc[result.x > 0.5].copy()
    return AllocationResult(selected, float(selected.utility.sum()), float(selected.cost.sum()), n, "optimal")


def zero_sum_equilibrium(values):
    """Both players' minimax strategies, plus a numerical duality certificate."""
    a = finite_matrix(values, max_size=400)
    m, n = a.shape
    row = linprog(np.r_[np.zeros(m), -1.0],
                  A_ub=np.column_stack((-a.T, np.ones(n))), b_ub=np.zeros(n),
                  A_eq=np.array([np.r_[np.ones(m), 0.0]]), b_eq=[1.0],
                  bounds=[(0, 1)] * m + [(None, None)], method="highs")
    col = linprog(np.r_[np.zeros(n), 1.0],
                  A_ub=np.column_stack((a, -np.ones(m))), b_ub=np.zeros(m),
                  A_eq=np.array([np.r_[np.ones(n), 0.0]]), b_eq=[1.0],
                  bounds=[(0, 1)] * n + [(None, None)], method="highs")
    if not row.success or not col.success:
        raise ValueError("均衡求解失败，请检查收益矩阵。")
    p, q = row.x[:-1], col.x[:-1]
    lower, upper = float(np.min(p @ a)), float(np.max(a @ q))
    return {"row_strategy": p, "column_strategy": q,
            "value": float(p @ a @ q), "duality_gap": max(0.0, upper - lower),
            "pure_maximin": float(a.min(axis=1).max()),
            "pure_minimax": float(a.max(axis=0).min())}


def pure_nash_equilibria(row_payoffs, column_payoffs):
    """All pure mutual best responses, including ties; not all mixed equilibria."""
    a, b = finite_matrix(row_payoffs, 400), finite_matrix(column_payoffs, 400)
    if a.shape != b.shape:
        raise ValueError("双方收益矩阵形状必须相同。")
    best_row = np.isclose(a, a.max(axis=0), rtol=0, atol=1e-9)
    best_col = np.isclose(b, b.max(axis=1)[:, None], rtol=0, atol=1e-9)
    return [tuple(map(int, location)) for location in np.argwhere(best_row & best_col)]
