__author__ = "Dong Yihan"

import matplotlib.pyplot as plt
import json
import math
import numpy as np


def draw_line_charts_no_concession_general_trend(result_dir, beta, agent_type):
    x_ray = list(range(0, beta + 1))
    y_ray_utility_1_agent1 = [2.0]
    y_ray_utility_2_agent1 = [2.0]
    y_ray_utility_1_agent2 = [2.0]
    y_ray_utility_2_agent2 = [2.0]
    y_ray_utility_1_agent3 = [-2.0]
    y_ray_utility_2_agent3 = [-2.0]
    y_ray_utility_1_agent4 = [-2.0]
    y_ray_utility_2_agent4 = [-2.0]
    for i in range(1, beta + 1):
        data_dir = result_dir + f"beta{i}.json"
        with open(data_dir, "r") as data_file:
            data = json.load(data_file)
        y_ray_utility_1_agent1.append(data["agent_1_utility"]["utility_1"][-1])
        y_ray_utility_2_agent1.append(data["agent_1_utility"]["utility_2"][-1])
        y_ray_utility_1_agent2.append(data["agent_2_utility"]["utility_1"][-1])
        y_ray_utility_2_agent2.append(data["agent_2_utility"]["utility_2"][-1])
        y_ray_utility_1_agent3.append(data["agent_3_utility"]["utility_1"][-1])
        y_ray_utility_2_agent3.append(data["agent_3_utility"]["utility_2"][-1])
        y_ray_utility_1_agent4.append(data["agent_4_utility"]["utility_1"][-1])
        y_ray_utility_2_agent4.append(data["agent_4_utility"]["utility_2"][-1])

    plt.figure(figsize=(12, 6))
    plt.xlabel("Acceptance Parameter Beta")
    plt.ylabel("Utilities of Agents")
    plt.title(f"Utility Adjustments of {agent_type}")
    plt.xticks(x_ray)
    plt.plot(x_ray, y_ray_utility_1_agent1, "r-", label="Agent 1 & high significance")
    plt.plot(x_ray, y_ray_utility_2_agent1, "r--", label="Agent 1 & low significance")
    plt.plot(x_ray, y_ray_utility_1_agent2, "b-", label="Agent 2 & high significance")
    plt.plot(x_ray, y_ray_utility_2_agent2, "b--", label="Agent 2 & low significance")
    plt.plot(x_ray, y_ray_utility_1_agent3, "m-", label="Agent 3 & high significance")
    plt.plot(x_ray, y_ray_utility_2_agent3, "m--", label="Agent 3 & low significance")
    plt.plot(x_ray, y_ray_utility_1_agent4, "g-", label="Agent 4 & high significance")
    plt.plot(x_ray, y_ray_utility_2_agent4, "g--", label="Agent 4 & low significance")
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


def draw_line_charts_no_concession_specific_divided_behaviour():
    group1_2_results_dir = "./experiments_results/no_concession/group_1_2/"
    group2_2_results_dir = "./experiments_results/no_concession/group_2_2/"
    group3_2_results_dir = "./experiments_results/no_concession/group_3_2/"

    x_ray = list(range(0, 21))
    y_ray_agent_type_1 = [0.0]
    y_ray_agent_type_2 = [0.0]
    y_ray_agent_type_3 = [0.0]
    for i in range(1, 21):
        data1_dir = group1_2_results_dir + f"beta{i}.json"
        data2_dir = group2_2_results_dir + f"beta{i}.json"
        data3_dir = group3_2_results_dir + f"beta{i}.json"
        with open(data1_dir, "r") as data1_file:
            data1 = json.load(data1_file)
        with open(data2_dir, "r") as data2_file:
            data2 = json.load(data2_file)
        with open(data3_dir, "r") as data3_file:
            data3 = json.load(data3_file)
        y_ray_agent_type_1.append(data1["agent_4_utility"]["utility_3"][-1])
        y_ray_agent_type_2.append(data2["agent_4_utility"]["utility_3"][-1])
        y_ray_agent_type_3.append(data3["agent_4_utility"]["utility_3"][-1])

    plt.figure(figsize=(12, 6))
    plt.xlabel("Acceptance Parameter Beta")
    plt.ylabel("Utilities towards Issue 3")
    plt.title("Utility Adjustments towards Issue 3 with a Medium Significance Weight and Other Agents Retain")
    plt.xticks(x_ray)
    plt.plot(x_ray, y_ray_agent_type_1, "r-", label="Mechanism-Bounded Utility Agent")
    plt.plot(x_ray, y_ray_agent_type_2, "b-", label="Self-Regulating Utility Agent")
    plt.plot(x_ray, y_ray_agent_type_3, "m-", label="Self-Regulating Linguistic Agent")
    plt.tight_layout()
    plt.legend()
    plt.show()

    return


def draw_line_charts_with_concession_general_trend_step_by_step(result_dir, agent_type):
    data_dir = result_dir + "beta1.json"
    with open(data_dir, "r") as experiment_result_file:
        result_data = json.load(experiment_result_file)
    x_total = len(result_data["agent_1_utility"]["utility_1"])
    x_ray = list(range(0, x_total))
    y_ray_utility_1_agent_1 = result_data["agent_1_utility"]["utility_1"]  # high significance
    y_ray_utility_2_agent_1 = result_data["agent_1_utility"]["utility_2"]  # medium significance
    y_ray_utility_3_agent_1 = result_data["agent_1_utility"]["utility_3"]  # low significance
    y_ray_utility_1_agent_2 = result_data["agent_2_utility"]["utility_1"]  # low significance
    y_ray_utility_2_agent_2 = result_data["agent_2_utility"]["utility_2"]  # high significance
    y_ray_utility_3_agent_2 = result_data["agent_2_utility"]["utility_3"]  # medium significance
    y_ray_utility_1_agent_3 = result_data["agent_3_utility"]["utility_1"]  # medium significance
    y_ray_utility_2_agent_3 = result_data["agent_3_utility"]["utility_2"]  # low significance
    y_ray_utility_3_agent_3 = result_data["agent_3_utility"]["utility_3"]  # high significance

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Utility Adjustment of {agent_type}", fontsize=14)

    axs[0].plot(x_ray, y_ray_utility_1_agent_1, "r-", label="Agent 1 with High Significance")
    axs[0].plot(x_ray, y_ray_utility_1_agent_2, "b:", label="Agent 2 with Low Significance")
    axs[0].plot(x_ray, y_ray_utility_1_agent_3, "g--", label="Agent 3 with Medium Significance")
    axs[0].set_ylabel("Issue 1 Utilities")
    axs[0].legend()

    axs[1].plot(x_ray, y_ray_utility_2_agent_1, "r--", label="Agent 1 with Medium Significance")
    axs[1].plot(x_ray, y_ray_utility_2_agent_2, "b-", label="Agent 2 with High Significance")
    axs[1].plot(x_ray, y_ray_utility_2_agent_3, "g:", label="Agent 3 with Low Significance")
    axs[1].set_ylabel("Issue 2 Utilities")
    axs[1].legend()

    axs[2].plot(x_ray, y_ray_utility_3_agent_1, "r:", label="Agent 1 with Low Significance")
    axs[2].plot(x_ray, y_ray_utility_3_agent_2, "b--", label="Agent 2 with Medium Significance")
    axs[2].plot(x_ray, y_ray_utility_3_agent_3, "g-", label="Agent 3 with High Significance")
    axs[2].set_ylabel("Issue 3 Utilities")
    axs[2].set_xlabel("Turns of Negotiation")
    axs[2].legend()

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    plt.tight_layout()
    plt.show()

    return


def draw_line_charts_with_concession_calculating_absolute_loss(agent1_result_dir, agent2_result_dir, agent3_result_dir):
    """
    comparison among all kinds of agents and calculating the loss of them
    no need for parameters to pass agent names
    :return:
    """
    agent1_data_dir = agent1_result_dir + "beta1.json"
    agent2_data_dir = agent2_result_dir + "beta1.json"
    agent3_data_dir = agent3_result_dir + "beta1.json"

    with open(agent1_data_dir, "r") as f1:
        agent1_data = json.load(f1)
    with open(agent2_data_dir, "r") as f2:
        agent2_data = json.load(f2)
    with open(agent3_data_dir, "r") as f3:
        agent3_data = json.load(f3)
    x_total = len(agent1_data["agent_1_utility"]["utility_1"])
    x_ray = list(range(0, x_total))

    data_list = [agent1_data, agent2_data, agent3_data]
    # store step-by-step significance-weight loss for each kind of agent, 2-dimensional array
    swl_list = []
    for data in data_list:
        utility_1_agent_1 = data["agent_1_utility"]["utility_1"]  # high significance
        utility_2_agent_1 = data["agent_1_utility"]["utility_2"]  # medium significance
        utility_3_agent_1 = data["agent_1_utility"]["utility_3"]  # low significance
        utility_1_agent_2 = data["agent_2_utility"]["utility_1"]  # low significance
        utility_2_agent_2 = data["agent_2_utility"]["utility_2"]  # high significance
        utility_3_agent_2 = data["agent_2_utility"]["utility_3"]  # medium significance
        utility_1_agent_3 = data["agent_3_utility"]["utility_1"]  # medium significance
        utility_2_agent_3 = data["agent_3_utility"]["utility_2"]  # low significance
        utility_3_agent_3 = data["agent_3_utility"]["utility_3"]  # high significance
        high_significance = 0.9
        medium_significance = 0.5
        low_significance = 0.05
        separate_swl = []
        for i in range(0, x_total):
            print(i)
            loss_utility_1_agent_1 = high_significance * (abs(utility_1_agent_1[i] - utility_1_agent_1[0]))
            loss_utility_2_agent_1 = medium_significance * (abs(utility_2_agent_1[i] - utility_2_agent_1[0]))
            loss_utility_3_agent_1 = low_significance * (abs(utility_3_agent_1[i] - utility_3_agent_1[0]))
            loss_utility_1_agent_2 = low_significance * (abs(utility_1_agent_2[i] - utility_1_agent_2[0]))
            loss_utility_2_agent_2 = high_significance * (abs(utility_2_agent_2[i] - utility_2_agent_2[0]))
            loss_utility_3_agent_2 = medium_significance * (abs(utility_3_agent_2[i] - utility_3_agent_2[0]))
            loss_utility_1_agent_3 = medium_significance * (abs(utility_1_agent_3[i] - utility_1_agent_3[0]))
            loss_utility_2_agent_3 = low_significance * (abs(utility_2_agent_3[i] - utility_2_agent_3[0]))
            loss_utility_3_agent_3 = high_significance * (abs(utility_3_agent_3[i] - utility_3_agent_3[0]))
            significance_weight_loss = (loss_utility_1_agent_1 + loss_utility_2_agent_1 + loss_utility_3_agent_1 +
                                        loss_utility_1_agent_2 + loss_utility_2_agent_2 + loss_utility_3_agent_2 +
                                        loss_utility_1_agent_3 + loss_utility_2_agent_3 + loss_utility_3_agent_3)
            separate_swl.append(significance_weight_loss)
        swl_list.append(separate_swl)

    plt.figure(figsize=(12, 6))
    plt.xlabel("Turns of Discussion")
    plt.ylabel("Absolute Cost")
    plt.title("Step-by-step Absolute Cost of Each Type of Agent")
    plt.xticks(x_ray)
    plt.plot(x_ray, swl_list[0], "r-", label="Mechanism-Bounded Utility Agent")
    plt.plot(x_ray, swl_list[1], "b-", label="Self-Regulating Utility Agent")
    plt.plot(x_ray, swl_list[2], "m-", label="Self-Regulating Linguistic Agent")
    plt.tight_layout()
    plt.legend()
    plt.show()

    return


def draw_line_charts_with_concession_calculating_squared_loss(agent1_result_dir, agent2_result_dir, agent3_result_dir):
    """
    comparison among all kinds of agents and calculating the loss of them
    no need for parameters to pass agent names
    :return:
    """
    agent1_data_dir = agent1_result_dir + "beta1.json"
    agent2_data_dir = agent2_result_dir + "beta1.json"
    agent3_data_dir = agent3_result_dir + "beta1.json"

    with open(agent1_data_dir, "r") as f1:
        agent1_data = json.load(f1)
    with open(agent2_data_dir, "r") as f2:
        agent2_data = json.load(f2)
    with open(agent3_data_dir, "r") as f3:
        agent3_data = json.load(f3)
    x_total = len(agent1_data["agent_1_utility"]["utility_1"])
    x_ray = list(range(0, x_total))

    data_list = [agent1_data, agent2_data, agent3_data]
    # store step-by-step significance-weight loss for each kind of agent, 2-dimensional array
    swl_list = []
    for data in data_list:
        utility_1_agent_1 = data["agent_1_utility"]["utility_1"]  # high significance
        utility_2_agent_1 = data["agent_1_utility"]["utility_2"]  # medium significance
        utility_3_agent_1 = data["agent_1_utility"]["utility_3"]  # low significance
        utility_1_agent_2 = data["agent_2_utility"]["utility_1"]  # low significance
        utility_2_agent_2 = data["agent_2_utility"]["utility_2"]  # high significance
        utility_3_agent_2 = data["agent_2_utility"]["utility_3"]  # medium significance
        utility_1_agent_3 = data["agent_3_utility"]["utility_1"]  # medium significance
        utility_2_agent_3 = data["agent_3_utility"]["utility_2"]  # low significance
        utility_3_agent_3 = data["agent_3_utility"]["utility_3"]  # high significance
        high_significance = 0.9
        medium_significance = 0.5
        low_significance = 0.05
        separate_swl = []
        for i in range(0, x_total):
            print(i)
            loss_utility_1_agent_1 = high_significance * (math.pow(utility_1_agent_1[i] - utility_1_agent_1[0], 2))
            loss_utility_2_agent_1 = medium_significance * (math.pow(utility_2_agent_1[i] - utility_2_agent_1[0], 2))
            loss_utility_3_agent_1 = low_significance * (math.pow(utility_3_agent_1[i] - utility_3_agent_1[0], 2))
            loss_utility_1_agent_2 = low_significance * (math.pow(utility_1_agent_2[i] - utility_1_agent_2[0], 2))
            loss_utility_2_agent_2 = high_significance * (math.pow(utility_2_agent_2[i] - utility_2_agent_2[0], 2))
            loss_utility_3_agent_2 = medium_significance * (math.pow(utility_3_agent_2[i] - utility_3_agent_2[0], 2))
            loss_utility_1_agent_3 = medium_significance * (math.pow(utility_1_agent_3[i] - utility_1_agent_3[0], 2))
            loss_utility_2_agent_3 = low_significance * (math.pow(utility_2_agent_3[i] - utility_2_agent_3[0], 2))
            loss_utility_3_agent_3 = high_significance * (math.pow(utility_3_agent_3[i] - utility_3_agent_3[0], 2))
            significance_weight_loss = (loss_utility_1_agent_1 + loss_utility_2_agent_1 + loss_utility_3_agent_1 +
                                        loss_utility_1_agent_2 + loss_utility_2_agent_2 + loss_utility_3_agent_2 +
                                        loss_utility_1_agent_3 + loss_utility_2_agent_3 + loss_utility_3_agent_3)
            separate_swl.append(significance_weight_loss)
        swl_list.append(separate_swl)

    plt.figure(figsize=(12, 6))
    plt.xlabel("Turns of Discussion")
    plt.ylabel("Squared Cost")
    plt.title("Step-by-step Squared Cost of Each Type of Agent")
    plt.xticks(x_ray)
    plt.plot(x_ray, swl_list[0], "r-", label="Mechanism-Bounded Utility Agent")
    plt.plot(x_ray, swl_list[1], "b-", label="Self-Regulating Utility Agent")
    plt.plot(x_ray, swl_list[2], "m-", label="Self-Regulating Linguistic Agent")
    plt.tight_layout()
    plt.legend()
    plt.show()

    return


def huber(r, delta):
    a = abs(r)
    if a <= delta:
        return 0.5 * r * r
    else:
        return delta * (a - 0.5 * delta)


def draw_line_charts_with_concession_calculating_huber_loss(agent1_result_dir, agent2_result_dir, agent3_result_dir,
                                                            delta=0.4):
    """
    comparison among all kinds of agents and calculating the loss of them
    no need for parameters to pass agent names
    :return:
    """
    agent1_data_dir = agent1_result_dir + "beta1.json"
    agent2_data_dir = agent2_result_dir + "beta1.json"
    agent3_data_dir = agent3_result_dir + "beta1.json"

    with open(agent1_data_dir, "r") as f1:
        agent1_data = json.load(f1)
    with open(agent2_data_dir, "r") as f2:
        agent2_data = json.load(f2)
    with open(agent3_data_dir, "r") as f3:
        agent3_data = json.load(f3)
    x_total = len(agent1_data["agent_1_utility"]["utility_1"])
    x_ray = list(range(0, x_total))

    data_list = [agent1_data, agent2_data, agent3_data]
    # store step-by-step significance-weight loss for each kind of agent, 2-dimensional array
    huber_list = []
    for data in data_list:
        utility_1_agent_1 = data["agent_1_utility"]["utility_1"]  # high significance
        utility_2_agent_1 = data["agent_1_utility"]["utility_2"]  # medium significance
        utility_3_agent_1 = data["agent_1_utility"]["utility_3"]  # low significance
        utility_1_agent_2 = data["agent_2_utility"]["utility_1"]  # low significance
        utility_2_agent_2 = data["agent_2_utility"]["utility_2"]  # high significance
        utility_3_agent_2 = data["agent_2_utility"]["utility_3"]  # medium significance
        utility_1_agent_3 = data["agent_3_utility"]["utility_1"]  # medium significance
        utility_2_agent_3 = data["agent_3_utility"]["utility_2"]  # low significance
        utility_3_agent_3 = data["agent_3_utility"]["utility_3"]  # high significance
        u1a1_0, u2a1_0, u3a1_0 = utility_1_agent_1[0], utility_2_agent_1[0], utility_3_agent_1[0]
        u1a2_0, u2a2_0, u3a2_0 = utility_1_agent_2[0], utility_2_agent_2[0], utility_3_agent_2[0]
        u1a3_0, u2a3_0, u3a3_0 = utility_1_agent_3[0], utility_2_agent_3[0], utility_3_agent_3[0]
        high_significance = 0.9
        medium_significance = 0.5
        low_significance = 0.05
        separate_huber = []
        for i in range(0, x_total):
            print(i)
            r_u1a1 = utility_1_agent_1[i] - u1a1_0
            r_u2a1 = utility_2_agent_1[i] - u2a1_0
            r_u3a1 = utility_3_agent_1[i] - u3a1_0

            r_u1a2 = utility_1_agent_2[i] - u1a2_0
            r_u2a2 = utility_2_agent_2[i] - u2a2_0
            r_u3a2 = utility_3_agent_2[i] - u3a2_0

            r_u1a3 = utility_1_agent_3[i] - u1a3_0
            r_u2a3 = utility_3_agent_3[i] - u2a3_0
            r_u3a3 = utility_3_agent_3[i] - u3a3_0

            L = 0.0
            L += high_significance * huber(r_u1a1, delta)
            L += medium_significance * huber(r_u2a1, delta)
            L += low_significance * huber(r_u3a1, delta)

            L += low_significance * huber(r_u1a2, delta)
            L += high_significance * huber(r_u2a2, delta)
            L += medium_significance * huber(r_u3a2, delta)

            L += medium_significance * huber(r_u1a3, delta)
            L += low_significance * huber(r_u2a3, delta)
            L += high_significance * huber(r_u3a3, delta)

            separate_huber.append(L)

        huber_list.append(separate_huber)

    plt.figure(figsize=(12, 6))
    plt.xlabel("Turns of Discussion")
    plt.ylabel("Huber Cost")
    plt.title("Step-by-step Huber Cost of Each Type of Agent")
    plt.xticks(x_ray)
    plt.plot(x_ray, huber_list[0], "r-", label="Mechanism-Bounded Utility Agent")
    plt.plot(x_ray, huber_list[1], "b-", label="Self-Regulating Utility Agent")
    plt.plot(x_ray, huber_list[2], "m-", label="Self-Regulating Linguistic Agent")
    plt.tight_layout()
    plt.legend()
    plt.show()

    return


def draw_minimax_change(
        agent1_result_dir, agent2_result_dir, agent3_result_dir,
        normalize=False, plot_kind="bar"  # plot_kind: "bar" or "line"
):
    """
    Minimax Change (attitude-only, significance-weighted).
    For each agent type (mechanism, utility, linguistic), at each round t:
        L_i(t) = sum_k s_i^k * |u_i^k(t) - u_i^k(0)|
        MM(t)  = max_i L_i(t)           # minimax change at round t
    If normalize=True, each L_i(t) is divided by sum_k s_i^k BEFORE the max.

    plot_kind="bar": plot final-round minimax for the three agent types.
    plot_kind="line": plot step-by-step minimax curves.
    Returns: dict with per-method series and the final values.
    """

    agent1_data_dir = agent1_result_dir + "beta1.json"  # Mechanism-Bounded Utility Agent
    agent2_data_dir = agent2_result_dir + "beta1.json"  # Self-Regulating Utility Agent
    agent3_data_dir = agent3_result_dir + "beta1.json"  # Self-Regulating Linguistic Agent

    with open(agent1_data_dir, "r") as f1:
        mech_data = json.load(f1)
    with open(agent2_data_dir, "r") as f2:
        util_data = json.load(f2)
    with open(agent3_data_dir, "r") as f3:
        ling_data = json.load(f3)

    x_total = len(mech_data["agent_1_utility"]["utility_1"])
    x_ray = list(range(x_total))

    # Significance weights (your mapping)
    high, med, low = 0.9, 0.5, 0.05

    def series_from_data(data):
        """Compute per-round minimax series for one agent family."""
        # Utilities over time
        u1a1 = data["agent_1_utility"]["utility_1"]  # A1 issue1 (high)
        u2a1 = data["agent_1_utility"]["utility_2"]  # A1 issue2 (med)
        u3a1 = data["agent_1_utility"]["utility_3"]  # A1 issue3 (low)

        u1a2 = data["agent_2_utility"]["utility_1"]  # A2 issue1 (low)
        u2a2 = data["agent_2_utility"]["utility_2"]  # A2 issue2 (high)
        u3a2 = data["agent_2_utility"]["utility_3"]  # A2 issue3 (med)

        u1a3 = data["agent_3_utility"]["utility_1"]  # A3 issue1 (med)
        u2a3 = data["agent_3_utility"]["utility_2"]  # A3 issue2 (low)
        u3a3 = data["agent_3_utility"]["utility_3"]  # A3 issue3 (high)

        # Initial attitudes
        u1a1_0, u2a1_0, u3a1_0 = u1a1[0], u2a1[0], u3a1[0]
        u1a2_0, u2a2_0, u3a2_0 = u1a2[0], u2a2[0], u3a2[0]
        u1a3_0, u2a3_0, u3a3_0 = u1a3[0], u2a3[0], u3a3[0]

        # Per-agent significance totals (for optional normalization)
        sig_sum_A1 = high + med + low
        sig_sum_A2 = low + high + med
        sig_sum_A3 = med + low + high

        mm_series = []
        for t in range(x_total):
            # per-agent total weighted absolute change
            L_A1 = (high * abs(u1a1[t] - u1a1_0)
                    + med * abs(u2a1[t] - u2a1_0)
                    + low * abs(u3a1[t] - u3a1_0))
            L_A2 = (low * abs(u1a2[t] - u1a2_0)
                    + high * abs(u2a2[t] - u2a2_0)
                    + med * abs(u3a2[t] - u3a2_0))
            L_A3 = (med * abs(u1a3[t] - u1a3_0)
                    + low * abs(u2a3[t] - u2a3_0)
                    + high * abs(u3a3[t] - u3a3_0))

            if normalize:
                L_A1 /= sig_sum_A1
                L_A2 /= sig_sum_A2
                L_A3 /= sig_sum_A3

            mm_series.append(max(L_A1, L_A2, L_A3))
        return mm_series

    mm_mech = series_from_data(mech_data)
    mm_util = series_from_data(util_data)
    mm_ling = series_from_data(ling_data)

    # Plot
    if plot_kind == "line":
        plt.figure(figsize=(12, 6))
        plt.xlabel("Turns of Discussion")
        ylabel = "Minimax Change" + (" (normalized)" if normalize else "")
        plt.ylabel(ylabel)
        plt.title("Step-by-step Minimax Change of Each Type of Agent")
        plt.xticks(x_ray)
        plt.plot(x_ray, mm_mech, "r-", label="Mechanism-Bounded Utility Agent")
        plt.plot(x_ray, mm_util, "b-", label="Self-Regulating Utility Agent")
        plt.plot(x_ray, mm_ling, "m-", label="Self-Regulating Linguistic Agent")
        plt.tight_layout()
        plt.legend()
        plt.show()
    else:
        # final-round bar (compact, good for paper)
        finals = [mm_mech[-1], mm_util[-1], mm_ling[-1]]
        labels = ["Mechanism-Bounded Utility", "Self-Reg Utility", "Self-Reg Linguistic"]
        plt.figure(figsize=(8, 5))
        plt.title("Final-Round Minimax Change")
        ylabel = "Minimax Change" + (" (normalized)" if normalize else "")
        plt.ylabel(ylabel)
        bars = plt.bar(labels, finals)
        # Optional: annotate values on bars
        for b, v in zip(bars, finals):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01, f"{v:.3f}",
                     ha='center', va='bottom')
        plt.tight_layout();
        plt.show()

    return {
        "mech_series": mm_mech,
        "util_series": mm_util,
        "ling_series": mm_ling,
        "finals": {
            "mechanism": mm_mech[-1],
            "utility": mm_util[-1],
            "linguistic": mm_ling[-1]
        }
    }


def compare_proportional_compromise_final(
        agent1_result_dir, agent2_result_dir, agent3_result_dir,
        budget_rule="one_minus_s", plot=True
):
    """
    Proportional Compromise at the final round.
    For each family (mechanism, self-reg utility, self-reg linguistic), compute:
        L_i = sum_k s_i^k * |u_i^k(T) - u_i^k(0)|
        B_i = sum_k (1 - s_i^k)           [default budget_rule="one_minus_s"]
        λ = max_i ( L_i / B_i )

    Returns a dict with λ per family and the per-agent contributions.
    Set plot=False if you don’t want the bar chart.
    """

    # file paths
    mech_fp = agent1_result_dir + "beta1.json"  # Mechanism-Bounded Utility Agent
    util_fp = agent2_result_dir + "beta1.json"  # Self-Regulating Utility Agent
    ling_fp = agent3_result_dir + "beta1.json"  # Self-Regulating Linguistic Agent

    with open(mech_fp, "r") as f:
        mech = json.load(f)
    with open(util_fp, "r") as f:
        util = json.load(f)
    with open(ling_fp, "r") as f:
        ling = json.load(f)

    # significance mapping (same as your earlier code)
    high, med, low = 0.9, 0.5, 0.05

    # budgets per agent (depends only on s)
    def budgets():
        B1 = (1 - high) + (1 - med) + (1 - low)
        B2 = (1 - low) + (1 - high) + (1 - med)
        B3 = (1 - med) + (1 - low) + (1 - high)
        if budget_rule == "sum_s":  # optional alternative
            B1, B2, B3 = (high + med + low, low + high + med, med + low + high)
        return B1, B2, B3

    def lambda_for_family(data):
        # utilities over time
        u1a1 = data["agent_1_utility"]["utility_1"]  # A1 issue1 (high)
        u2a1 = data["agent_1_utility"]["utility_2"]  # A1 issue2 (med)
        u3a1 = data["agent_1_utility"]["utility_3"]  # A1 issue3 (low)

        u1a2 = data["agent_2_utility"]["utility_1"]  # A2 issue1 (low)
        u2a2 = data["agent_2_utility"]["utility_2"]  # A2 issue2 (high)
        u3a2 = data["agent_2_utility"]["utility_3"]  # A2 issue3 (med)

        u1a3 = data["agent_3_utility"]["utility_1"]  # A3 issue1 (med)
        u2a3 = data["agent_3_utility"]["utility_2"]  # A3 issue2 (low)
        u3a3 = data["agent_3_utility"]["utility_3"]  # A3 issue3 (high)

        T = len(u1a1) - 1  # final round index

        # initial
        u1a1_0, u2a1_0, u3a1_0 = u1a1[0], u2a1[0], u3a1[0]
        u1a2_0, u2a2_0, u3a2_0 = u1a2[0], u2a2[0], u3a2[0]
        u1a3_0, u2a3_0, u3a3_0 = u1a3[0], u2a3[0], u3a3[0]

        # per-agent weighted absolute change at final round
        L1 = (high * abs(u1a1[T] - u1a1_0) +
              med * abs(u2a1[T] - u2a1_0) +
              low * abs(u3a1[T] - u3a1_0))
        L2 = (low * abs(u1a2[T] - u1a2_0) +
              high * abs(u2a2[T] - u2a2_0) +
              med * abs(u3a2[T] - u3a2_0))
        L3 = (med * abs(u1a3[T] - u1a3_0) +
              low * abs(u2a3[T] - u2a3_0) +
              high * abs(u3a3[T] - u3a3_0))

        B1, B2, B3 = budgets()
        lam1, lam2, lam3 = L1 / B1, L2 / B2, L3 / B3
        lam = max(lam1, lam2, lam3)
        argmax = 1 if lam == lam1 else (2 if lam == lam2 else 3)
        return lam, (lam1, lam2, lam3), (L1, L2, L3), (B1, B2, B3), argmax

    lam_mech, perA_mech, L_mech, B_mech, who_mech = lambda_for_family(mech)
    lam_util, perA_util, L_util, B_util, who_util = lambda_for_family(util)
    lam_ling, perA_ling, L_ling, B_ling, who_ling = lambda_for_family(ling)

    results = {
        "lambda": {
            "mechanism": lam_mech,
            "utility": lam_util,
            "linguistic": lam_ling,
        },
        "per_agent_lambda": {
            "mechanism": {"A1": perA_mech[0], "A2": perA_mech[1], "A3": perA_mech[2], "worst_off": f"A{who_mech}"},
            "utility": {"A1": perA_util[0], "A2": perA_util[1], "A3": perA_util[2], "worst_off": f"A{who_util}"},
            "linguistic": {"A1": perA_ling[0], "A2": perA_ling[1], "A3": perA_ling[2], "worst_off": f"A{who_ling}"},
        },
        "per_agent_L_and_B": {
            "mechanism": {"L": L_mech, "B": B_mech},
            "utility": {"L": L_util, "B": B_util},
            "linguistic": {"L": L_ling, "B": B_ling},
        },
        "budget_rule": budget_rule
    }

    if plot:
        labels = ["Mechanism-Bounded Utility", "Self-Reg Utility", "Self-Reg Linguistic"]
        vals = [lam_mech, lam_util, lam_ling]
        plt.figure(figsize=(8, 5))
        plt.title("Final-Round Proportional Compromise (λ)")
        plt.ylabel("λ (lower is better)")
        bars = plt.bar(labels, vals)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01, f"{v:.3f}",
                     ha="center", va="bottom")
        plt.tight_layout();
        plt.show()

    return results


def compare_reciprocity_index_final(
        agent1_result_dir, agent2_result_dir, agent3_result_dir,
        plot=True, return_components=False, label_offset_ratio=0.01
):
    """
    Reciprocity Index (RI) at the final round for three families:
      - Mechanism-Bounded Utility
      - Self-Regulating Utility
      - Self-Regulating Linguistic

    JSON layout is the same as in your previous code.
    Significance mapping:
      A1: (utility_1=high, utility_2=med, utility_3=low)
      A2: (utility_1=low,  utility_2=high, utility_3=med)
      A3: (utility_1=med,  utility_2=low,  utility_3=high)
    """

    paths = [
        agent1_result_dir + "beta1.json",  # Mechanism
        agent2_result_dir + "beta1.json",  # Self-Reg Utility
        agent3_result_dir + "beta1.json",  # Self-Reg Linguistic
    ]
    names = ["Mechanism-Bounded Utility", "Self-Reg Utility", "Self-Reg Linguistic"]

    datasets = [json.load(open(p, "r")) for p in paths]

    agents = ["agent_1", "agent_2", "agent_3"]
    issues = ["utility_1", "utility_2", "utility_3"]

    # significance weights
    high, med, low = 0.9, 0.5, 0.05
    s_map = {
        ("agent_1", "utility_1"): high,
        ("agent_1", "utility_2"): med,
        ("agent_1", "utility_3"): low,

        ("agent_2", "utility_1"): low,
        ("agent_2", "utility_2"): high,
        ("agent_2", "utility_3"): med,

        ("agent_3", "utility_1"): med,
        ("agent_3", "utility_2"): low,
        ("agent_3", "utility_3"): high,
    }

    def ri_for_family(data):
        # final round index
        T = len(data["agent_1_utility"]["utility_1"]) - 1

        # initial & final per (agent, issue)
        U0, UT = {}, {}
        for a in agents:
            for k in issues:
                seq = data[f"{a}_utility"][k]
                U0[(a, k)] = seq[0]
                UT[(a, k)] = seq[T]

        # precompute: for each issue k, avg significance of "others" and avg movement of "others"
        RI = {}
        components = {}  # optional return: (Give_i, Receive_i)
        for i in agents:
            Give_i, Receive_i = 0.0, 0.0
            for k in issues:
                # average significance of others on issue k
                s_bar_minus_i = np.mean([s_map[(j, k)] for j in agents if j != i])
                # average absolute movement of others on issue k
                delta_bar_minus_i = np.mean([abs(UT[(j, k)] - U0[(j, k)]) for j in agents if j != i])
                # i's own movement on issue k
                delta_i_k = abs(UT[(i, k)] - U0[(i, k)])

                Give_i += s_bar_minus_i * delta_i_k
                Receive_i += s_map[(i, k)] * delta_bar_minus_i

            RI[i] = Receive_i - Give_i
            components[i] = (Give_i, Receive_i)
        return RI, components

    all_RI = []
    all_comp = []
    for data in datasets:
        ri, comp = ri_for_family(data)
        all_RI.append(ri)
        all_comp.append(comp)

    # ---- Plot: grouped bars per agent (one group per agent, three bars = families) ----
    if plot:
        x = np.arange(len(agents))
        w = 0.28
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.set_title("Final-Round Reciprocity Index (RI)")
        ax.set_ylabel("RI  (Receive − Give)")
        ax.axhline(0, color="gray", linewidth=1)

        mech_vals = [all_RI[0][a] for a in agents]
        util_vals = [all_RI[1][a] for a in agents]
        ling_vals = [all_RI[2][a] for a in agents]

        bars1 = ax.bar(x - w, mech_vals, width=w, label=names[0])
        bars2 = ax.bar(x, util_vals, width=w, label=names[1])
        bars3 = ax.bar(x + w, ling_vals, width=w, label=names[2])

        ax.set_xticks(x, ["Agent 1", "Agent 2", "Agent 3"])
        ax.legend()

        # ---- tighter annotations: offset is 1% of current y-span ----
        y_min, y_max = ax.get_ylim()
        y_span = max(1e-9, y_max - y_min)  # avoid zero division
        dy = label_offset_ratio * y_span

        def annotate_bars(bars):
            for b in bars:
                v = b.get_height()
                # place label just outside bar tip
                y = v + (dy if v >= 0 else -dy)
                va = "bottom" if v >= 0 else "top"
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    y,
                    f"{v:.3f}",
                    ha="center", va=va,
                    fontsize=10
                )

        annotate_bars(bars1)
        annotate_bars(bars2)
        annotate_bars(bars3)

        # If labels still feel far/close, tweak label_offset_ratio (e.g., 0.005 or 0.015)
        plt.tight_layout();
        plt.show()

    # package results nicely
    results = {
        "RI_per_family": {
            names[0]: all_RI[0],
            names[1]: all_RI[1],
            names[2]: all_RI[2],
        },
        "summary": {
            names[0]: {
                "mean_RI": float(np.mean(list(all_RI[0].values()))),
                "min_RI": float(np.min(list(all_RI[0].values()))),
                "max_RI": float(np.max(list(all_RI[0].values())))
            },
            names[1]: {
                "mean_RI": float(np.mean(list(all_RI[1].values()))),
                "min_RI": float(np.min(list(all_RI[1].values()))),
                "max_RI": float(np.max(list(all_RI[1].values())))
            },
            names[2]: {
                "mean_RI": float(np.mean(list(all_RI[2].values()))),
                "min_RI": float(np.min(list(all_RI[2].values()))),
                "max_RI": float(np.max(list(all_RI[2].values())))
            }
        }
    }
    if return_components:
        results["components"] = {
            names[0]: all_comp[0],  # dict: agent -> (Give, Receive)
            names[1]: all_comp[1],
            names[2]: all_comp[2],
        }
    return results


def compute_ideal_utility_point(utilities: list[float], weights: list[float]) -> float:
    """
    Calculate the ideal utility value that minimizes the total significance-weighted loss.

    Args:
        utilities (list[float]): List of utility values from different agents.
        weights (list[float]): Corresponding significance weights for each utility.

    Returns:
        float: Ideal consensus utility value.
    """
    if len(utilities) != len(weights):
        raise ValueError("Length of utilities and weights must be equal.")
    if sum(weights) == 0:
        raise ValueError("Sum of weights cannot be zero.")

    weighted_sum = sum(w * u for u, w in zip(utilities, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight


def compute_weighted_median(utilities: list[float], weights: list[float]) -> float:
    """
    Compute the weighted median (minimizing significance-weighted absolute loss).
    """
    if len(utilities) != len(weights):
        raise ValueError("Length of utilities and weights must be equal.")

    # Sort by utility
    sorted_pairs = sorted(zip(utilities, weights), key=lambda x: x[0])
    utilities_sorted, weights_sorted = zip(*sorted_pairs)

    cumulative_weights = np.cumsum(weights_sorted)
    cutoff = sum(weights_sorted) / 2

    for i, cw in enumerate(cumulative_weights):
        if cw >= cutoff:
            return utilities_sorted[i]


if __name__ == "__main__":
    # graph for comparative results among different agents with high/low significance weights

    # group1_results_dir = "./experiments_results/no_concession/group_1/"
    # draw_line_charts_no_concession_general_trend(
    #     result_dir=group1_results_dir, beta=20, agent_type="Mechanism-Bounded Utility Agents")
    # group2_results_dir = "./experiments_results/no_concession/group_2/"
    # draw_line_charts_no_concession_general_trend(
    #     result_dir=group2_results_dir, beta=20, agent_type="Self-Regulating Utility Agents (CoT & Few-shot Learning)")
    # group3_results_dir = "./experiments_results/no_concession/group_3/"
    # draw_line_charts_no_concession_general_trend(
    #     result_dir=group3_results_dir, beta=20, agent_type="Self-Regulating Linguistic Agents (CoT and Few-shot Learning)")

    # graph for comparative results among different agents when other's attitudes separate from each other
    # draw_line_charts_no_concession_specific_divided_behaviour()

    # figures for comparative results among different agents with the concession strategy

    experiment_result_dir_1 = "experiments_results/with_concession/group_1_4.1_markup/"
    # draw_line_charts_with_concession_general_trend_step_by_step(
    #     result_dir=experiment_result_dir_1, agent_type="Mechanism-Bounded Utility Agents")
    experiment_result_dir_2 = "./experiments_results/with_concession/group_2_4.1_markup/"
    # draw_line_charts_with_concession_general_trend_step_by_step(
    #     result_dir=experiment_result_dir_2,
    #     agent_type="Self-Regulating Utility Agents")
    experiment_result_dir_3 = "./experiments_results/with_concession/group_3_4.1/"
    # draw_line_charts_with_concession_general_trend_step_by_step(
    #     result_dir=experiment_result_dir_3,
    #     agent_type="Self-Regulating Linguistic Agents")
    # draw_line_charts_with_concession_calculating_absolute_loss(agent1_result_dir=experiment_result_dir_1,
    #                                                            agent2_result_dir=experiment_result_dir_2,
    #                                                            agent3_result_dir=experiment_result_dir_3)
    # draw_line_charts_with_concession_calculating_squared_loss(agent1_result_dir=experiment_result_dir_1,
    #                                                           agent2_result_dir=experiment_result_dir_2,
    #                                                           agent3_result_dir=experiment_result_dir_3)
    # draw_line_charts_with_concession_calculating_huber_loss(agent1_result_dir=experiment_result_dir_1,
    #                                                         agent2_result_dir=experiment_result_dir_2,
    #                                                         agent3_result_dir=experiment_result_dir_3)

    # draw_minimax_change(agent1_result_dir=experiment_result_dir_1,
    #                     agent2_result_dir=experiment_result_dir_2,
    #                     agent3_result_dir=experiment_result_dir_3,)
    # draw_minimax_change(agent1_result_dir=experiment_result_dir_1,
    #                     agent2_result_dir=experiment_result_dir_2,
    #                     agent3_result_dir=experiment_result_dir_3,
    #                     plot_kind="line")

    # compare_proportional_compromise_final(agent1_result_dir=experiment_result_dir_1,
    #                                       agent2_result_dir=experiment_result_dir_2,
    #                                       agent3_result_dir=experiment_result_dir_3)

    compare_reciprocity_index_final(agent1_result_dir=experiment_result_dir_1,
                                    agent2_result_dir=experiment_result_dir_2,
                                    agent3_result_dir=experiment_result_dir_3)

    # utilities = [2.0, 0.0, -1.0]
    # significance_weights = [0.9, 0.5, 0.05]
    # ideal_point = compute_ideal_utility_point(utilities=utilities, weights=significance_weights)
    # print(ideal_point)
    # ideal_point_2 = compute_weighted_median(utilities=utilities, weights=significance_weights)
    # print(ideal_point_2)
