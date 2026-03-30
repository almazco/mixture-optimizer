from typing import Dict, Any, List
import numpy as np
from scipy.optimize import linprog


def optimize_mixture(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Основная функция оптимизации многокомпонентной смеси методом линейного программирования.

    Args:
        config (dict): Конфигурация задачи (total_mass и список components)

    Returns:
        dict: Результат оптимизации
    """
    total_mass = config["total_mass"]
    components: List[Dict] = config["components"]
    n = len(components)

    # Целевая функция — минимизация стоимости
    c = np.array([comp["cost"] for comp in components])

    # Ограничение равенства: сумма масс = total_mass
    A_eq = np.ones((1, n))
    b_eq = np.array([total_mass])

    # Ограничения-неравенства (нижние и верхние границы)
    A_ub: List[List[float]] = []
    b_ub: List[float] = []

    for i, comp in enumerate(components):
        if comp.get("min") is not None:
            A_ub.append([-1 if j == i else 0 for j in range(n)])
            b_ub.append(-comp["min"])
        if comp.get("max") is not None:
            A_ub.append([1 if j == i else 0 for j in range(n)])
            b_ub.append(comp["max"])

    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None

    # Решение задачи
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * n, method='highs')

    if not res.success:
        return {"error": "Задача не имеет решения. Проверьте ограничения."}

    # Формирование результата
    result = {
        "status": "success",
        "total_mass": total_mass,
        "total_cost": round(res.fun, 4),
        "optimal_composition": {
            comp["name"]: round(res.x[i], 2) for i, comp in enumerate(components)
        }
    }

    return result
