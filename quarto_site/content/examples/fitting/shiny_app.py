from shiny import App, reactive, render, ui
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
import warnings


def linear_model(x, a, b):
    return a * x + b


def exponential_model(x, amplitude, rate, baseline):
    return amplitude * np.exp(-rate * x) + baseline


def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c


def double_exponential_model(x, a1, rate1, a2, rate2, baseline):
    return a1 * np.exp(-rate1 * x) + a2 * np.exp(-rate2 * x) + baseline


MODEL_CONFIG = {
    "Linear": {
        "function": linear_model,
        "guess": lambda x, y: [0.0, float(np.mean(y))],
        "bounds": (-np.inf, np.inf),
        "parameters": ["slope", "intercept"],
    },
    "Single exponential": {
        "function": exponential_model,
        "guess": lambda x, y: [
            max(float(y[0] - y[-1]), 0.1),
            0.1,
            max(float(y[-1]), 0.0),
        ],
        "bounds": ([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
        "parameters": ["amplitude", "rate", "baseline"],
    },
    "Quadratic polynomial": {
        "function": quadratic_model,
        "guess": lambda x, y: list(np.polyfit(x, y, 2)),
        "bounds": (-np.inf, np.inf),
        "parameters": ["x²", "x", "constant"],
    },
    "Double exponential": {
        "function": double_exponential_model,
        "guess": lambda x, y: [
            max(float((y[0] - y[-1]) * 0.7), 0.1),
            0.15,
            max(float((y[0] - y[-1]) * 0.3), 0.1),
            0.03,
            max(float(y[-1]), 0.0),
        ],
        "bounds": ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]),
        "parameters": ["amplitude 1", "rate 1", "amplitude 2", "rate 2", "baseline"],
    },
}


def true_signal(x, pattern):
    if pattern == "Exponential decay":
        return 1.05 * np.exp(-0.12 * x) + 0.05
    if pattern == "Linear trend":
        return 0.95 - 0.035 * x
    return 1.0 - 0.035 * x + 0.0018 * (x - 10) ** 2


def simulate_data(pattern, n_points, noise, seed, resimulate):
    rng = np.random.default_rng(seed + 1009 * resimulate)
    x = np.linspace(0, 20, n_points)
    clean = true_signal(x, pattern)
    y = clean + rng.normal(0, noise, size=n_points)
    return x, y, clean


def fit_model(x, y, model_name):
    config = MODEL_CONFIG[model_name]
    function = config["function"]
    guess = config["guess"](x, y)
    bounds = config["bounds"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(
            function,
            x,
            y,
            p0=guess,
            bounds=bounds,
            maxfev=20000,
        )
    predicted = function(x, *popt)
    residuals = y - predicted
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - rss / tss if tss > 0 else np.nan
    return popt, predicted, residuals, rss, r_squared


app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "pattern",
                "True data pattern",
                ["Exponential decay", "Linear trend", "Curved trend"],
            ),
            ui.input_select(
                "model",
                "Fitted model",
                ["Linear", "Single exponential", "Quadratic polynomial", "Double exponential"],
                selected="Linear",
            ),
            ui.input_slider("noise", "Noise level", 0.0, 0.20, 0.04, step=0.01),
            ui.input_slider("n_points", "Number of points", 6, 24, 11, step=1),
            ui.input_slider("seed", "Random seed", 1, 100, 7, step=1),
            ui.input_checkbox("extrapolate", "Show extrapolation", False),
            ui.input_action_button("resimulate", "Resimulate"),
        ),
        ui.output_plot("fit_plot"),
        ui.output_table("metrics"),
    )
)


def server(input, output, session):
    @reactive.calc
    def data():
        return simulate_data(
            input.pattern(),
            input.n_points(),
            input.noise(),
            input.seed(),
            input.resimulate(),
        )

    @reactive.calc
    def fit():
        x, y, _ = data()
        return fit_model(x, y, input.model())

    @output
    @render.plot(alt="Fitted model and residual plot")
    def fit_plot():
        x, y, clean = data()
        popt, predicted, residuals, rss, r_squared = fit()
        config = MODEL_CONFIG[input.model()]

        x_max = (36 if input.extrapolate() else 20) + 0.5
        x_min = 0 - 0.5
        x_grid = np.linspace(x_min, x_max, 240)
        fitted_grid = config["function"](x_grid, *popt)
        true_grid = true_signal(x_grid, input.pattern())

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            gridspec_kw={"height_ratios": [2.2, 1]},
            sharex=False,
        )

        axes[0].scatter(x, y, color="#244c5a", s=42, label="observed data")
        axes[0].plot(x_grid, fitted_grid, color="#b84a39", linewidth=2.2, label="fitted model")
        axes[0].plot(
            x_grid,
            true_grid,
            color="#5e7f42",
            linewidth=1.7,
            linestyle="--",
            label="true pattern",
        )
        axes[0].axvspan(20, x_max, color="#eeeeee", alpha=0.8 if x_max > 20 else 0)
        axes[0].set(ylabel="Signal", xlim=(0, x_max))
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")
        axes[0].set_xlim(x_min, x_max)

        axes[1].axhline(0, color="#202020", linewidth=1)
        axes[1].scatter(x, residuals, color="#6f5a9b", s=38)
        axes[1].vlines(x, 0, residuals, color="#6f5a9b", alpha=0.45)
        axes[1].set(xlabel="Time", ylabel="Residual", xlim=(0, 20))
        axes[1].grid(alpha=0.25)
        axes[1].set_xlim(x_min, x_max)


        fig.tight_layout()
        return fig

    @output
    @render.table
    def metrics():
        popt, predicted, residuals, rss, r_squared = fit()
        config = MODEL_CONFIG[input.model()]
        parameter_text = ", ".join(
            f"{name}={value:.3g}"
            for name, value in zip(config["parameters"], popt)
        )
        return pd.DataFrame(
            {
                "Metric": [
                    "R²",
                    "Residual sum of squares",
                    "Model parameters",
                    "Fitted values",
                ],
                "Value": [
                    f"{r_squared:.4f}",
                    f"{rss:.4f}",
                    str(len(popt)),
                    parameter_text,
                ],
            }
        )

app = App(app_ui, server)