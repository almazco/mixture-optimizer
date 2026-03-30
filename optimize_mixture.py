import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime

def optimize_mixture(config):
    """
    Основная функция оптимизации многокомпонентной смеси
    config - словарь с параметрами (см. example_input.json)
    """
    total_mass = config['total_mass']
    components = config['components']
    
    n = len(components)
    
    # Целевая функция (стоимость)
    c = np.array([comp['cost'] for comp in components])
    
    # Ограничение равенства: сумма масс = total_mass
    A_eq = np.ones((1, n))
    b_eq = np.array([total_mass])
    
    # Ограничения-неравенства
    A_ub = []
    b_ub = []
    
    for i, comp in enumerate(components):
        # Нижняя граница
        if comp.get('min') is not None:
            A_ub.append([-1 if j == i else 0 for j in range(n)])
            b_ub.append(-comp['min'])
        # Верхняя граница
        if comp.get('max') is not None:
            A_ub.append([1 if j == i else 0 for j in range(n)])
            b_ub.append(comp['max'])
    
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    
    # Решение задачи
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * n, method='highs')
    
    if not res.success:
        return {"error": "Задача не имеет решения. Проверьте ограничения."}
    
    # Формирование результата
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_mass": total_mass,
        "optimal_composition": {},
        "total_cost": round(res.fun, 4),
        "status": "success"
    }
    
    for i, comp in enumerate(components):
        result["optimal_composition"][comp["name"]] = round(res.x[i], 2)
    
    # Сохранение графика
    plot_cost_sensitivity(components, total_mass, result)
    
    # Сохранение результата в CSV
    df = pd.DataFrame([result["optimal_composition"]])
    df["total_cost"] = result["total_cost"]
    df.to_csv("optimal_mixture_result.csv", index=False)
    
    return result


def plot_cost_sensitivity(components, total_mass, result):
    """Строит график зависимости стоимости от массы активного вещества"""
    plt.figure(figsize=(10, 6))
    active_name = next((comp["name"] for comp in components if comp.get("min") is not None), "A")
    
    masses = np.linspace(40, 120, 17)
    costs = []
    
    for m in masses:
        temp_config = {
            "total_mass": total_mass,
            "components": []
        }
        for comp in components:
            new_comp = comp.copy()
            if comp["name"] == active_name:
                new_comp["min"] = m
            temp_config["components"].append(new_comp)
        
        try:
            r = optimize_mixture(temp_config)
            if isinstance(r, dict) and "total_cost" in r:
                costs.append(r["total_cost"])
            else:
                costs.append(None)
        except:
            costs.append(None)
    
    plt.plot(masses, costs, marker='o', linewidth=2.5, color='#2ca02c')
    plt.xlabel('Минимальная масса активного вещества, мг')
    plt.ylabel('Минимальная стоимость смеси, у.е.')
    plt.title('Зависимость стоимости от содержания активного вещества')
    plt.grid(True, alpha=0.3)
    plt.savefig('cost_sensitivity_plot.png', dpi=400, bbox_inches='tight')
    plt.close()


# === Пример запуска ===
if __name__ == "__main__":
    with open("example_input.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    result = optimize_mixture(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
