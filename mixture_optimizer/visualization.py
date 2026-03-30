import matplotlib.pyplot as plt
import numpy as np
from .core import optimize_mixture


def plot_cost_sensitivity(config: dict, save_path: str = "cost_sensitivity.png"):
    """Строит график чувствительности стоимости к изменению минимальной массы активного вещества"""
    total_mass = config["total_mass"]
    active_comp = next((c for c in config["components"] if c.get("min") is not None), None)
    if not active_comp:
        return

    masses = np.linspace(40, 130, 19)
    costs = []

    for m in masses:
        temp_config = config.copy()
        for comp in temp_config["components"]:
            if comp["name"] == active_comp["name"]:
                comp["min"] = float(m)
        try:
            res = optimize_mixture(temp_config)
            costs.append(res.get("total_cost"))
        except:
            costs.append(None)

    plt.figure(figsize=(11, 6))
    plt.plot(masses, costs, marker='o', linewidth=2.5, color='#2ca02c')
    plt.xlabel('Минимальная масса активного вещества, мг')
    plt.ylabel('Минимальная стоимость смеси, у.е.')
    plt.title('Зависимость стоимости от содержания активного вещества')
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
