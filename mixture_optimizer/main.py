import json
from mixture_optimizer.core import optimize_mixture
from mixture_optimizer.visualization import plot_cost_sensitivity

if __name__ == "__main__":
    with open("example_input.json", encoding="utf-8") as f:
        config = json.load(f)

    result = optimize_mixture(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("status") == "success":
        plot_cost_sensitivity(config)
        print("График сохранён: cost_sensitivity.png")
