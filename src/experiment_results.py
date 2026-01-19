__author__ = "Dong Yihan"

import json

agent1_combination_dir = "./combination/agent1.json"
agent2_combination_dir = "./combination/agent2.json"
agent3_combination_dir = "./combination/agent3.json"
agent4_combination_dir = "./combination/agent4.json"


def extract_step_by_step_utilities(combination_dir):
    with open(combination_dir, "r") as f:
        combination_history = json.load(f)
    q1_history = [element["attitude"] for element in combination_history
                  if element["index"] == 1]
    q2_history = [element["attitude"] for element in combination_history
                  if element["index"] == 2]
    q3_history = [element["attitude"] for element in combination_history
                  if element["index"] == 3]

    utilities_history = {
        "utility_1": q1_history,
        "utility_2": q2_history,
        "utility_3": q3_history
    }

    return utilities_history


def extract_step_by_step_attitudes_to_utilities(combination_dir):
    with open(combination_dir, "r") as f:
        combination_history = json.load(f)

    opinion1_history = [element["attitude_description"] for element in combination_history if element["index"] == 1]
    opinion2_history = [element["attitude_description"] for element in combination_history if element["index"] == 2]
    opinion3_history = [element["attitude_description"] for element in combination_history if element["index"] == 3]

    utility1_history = convert_language_to_utility(attitudes_list=opinion1_history)
    utility2_history = convert_language_to_utility(attitudes_list=opinion2_history)
    utility3_history = convert_language_to_utility(attitudes_list=opinion3_history)

    attitude_history = {
        "attitude_1": opinion1_history,
        "attitude_2": opinion2_history,
        "attitude_3": opinion3_history,
        "utility_1": utility1_history,
        "utility_2": utility2_history,
        "utility_3": utility3_history
    }
    return attitude_history


def convert_language_to_utility(attitudes_list):
    """
    The reversing process of convertion of utility to language. Use this function after the experiment to check
     the results of discussions
    :param attitudes_list:
    :return:
    """

    # Base labels and their reference crisp values (center points)
    base_to_value = {
        "Strongly Disagree": -2.0,
        "Disagree": -1.2,
        "Probably Disagree": -0.9,
        "Slightly Disagree": -0.6,
        "Neutral": 0.0,
        "Slightly Agree": 0.6,
        "Probably Agree": 0.9,
        "Agree": 1.2,
        "Strongly Agree": 2.0
    }

    # Modifier offsets from the base value (tunable)
    modifier_offsets = {
        "Definitely": 0.1,
        "Completely": 0.1,
        "Probably": 0.2,
        "Somewhat between": 0.3,  # used with two bases, will take midpoint
        "Slightly leaning toward": 0.4
    }

    attitude_defuzzify_value = []

    for attitude_description in attitudes_list:
        if attitude_description.startswith("Somewhat between"):
            stripped = attitude_description.replace("Somewhat between", "").strip()
            if " and " in stripped:
                base1, base2 = stripped.split(" and ")
                for k in modifier_offsets.keys():
                    if k in base1:
                        base1 = base1.replace(k, "").strip()
                    if k in base2:
                        base2 = base2.replace(k, "").strip()
                print("base1:", base1)
                print("\nbase2:", base2)
                val1 = base_to_value[base1.strip()]
                val2 = base_to_value[base2.strip()]
                attitude_defuzzify_value.append(round((val1 + val2) / 2, 2))
        for modifier, offset in modifier_offsets.items():
            if attitude_description.startswith(modifier):
                base_phase = attitude_description.replace(modifier, "").strip()
                base_value = base_to_value.get(base_phase)
                if base_value is not None:
                    direction = 1 if base_value < 0 else -1 if base_value > 0 else 0
                    attitude_defuzzify_value.append(round(base_value + direction * offset, 2))

    return attitude_defuzzify_value


def convert_language_to_utility_results_for_utility3(result_dir):
    with open(result_dir, "r") as experiment_results_file:
        experiment_results = json.load(experiment_results_file)
    attitude_list = experiment_results["agent_4_utility"]["attitude_3"]
    attitude_defuzzify_value = convert_language_to_utility(attitudes_list=attitude_list)
    experiment_results["agent_4_utility"]["utility_3"] = attitude_defuzzify_value
    with open(result_dir, "w") as experiment_results_file_w:
        json.dump(experiment_results, experiment_results_file_w, indent=4)

    return


def convert_language_to_utility_with_concession(result_dir):
    with open(result_dir, "r") as experiments_results_file:
        experiments_results_temp = json.load(experiments_results_file)
    for i in range(1, 4):
        for j in range(1, 4):
            attitude_list = experiments_results_temp[f"agent_{i}_utility"][f"attitude_{j}"]
            corresponding_utility_list = convert_language_to_utility(attitudes_list=attitude_list)
            if len(corresponding_utility_list) != 11:
                print(len(corresponding_utility_list))
                print(f"agent {i} opinion {j}")
            experiments_results_temp[f"agent_{i}_utility"][f"utility_{j}"] = corresponding_utility_list

    with open("./experiments_results/with_concession/group_3_4.1/beta1_2.json", "w") as export_dir:
        json.dump(experiments_results_temp, export_dir, indent=4)


if __name__ == "__main__":
    # agent1_utility_history = extract_step_by_step_attitudes_to_utilities(combination_dir=agent1_combination_dir)
    # agent2_utility_history = extract_step_by_step_attitudes_to_utilities(combination_dir=agent2_combination_dir)
    # agent3_utility_history = extract_step_by_step_attitudes_to_utilities(combination_dir=agent3_combination_dir)
    # agent4_utility_history = extract_step_by_step_attitudes_to_utilities(combination_dir=agent4_combination_dir)

    experiments_results_dir = "experiments_results/with_concession_2/group_3_4.1/beta1.json"
    # convert_language_to_utility_results_for_utility3(result_dir=experiments_results_dir)
    convert_language_to_utility_with_concession(result_dir=experiments_results_dir)

    # experiments_results = {
    #     "agent_1_utility": agent1_utility_history,
    #     "agent_2_utility": agent2_utility_history,
    #     "agent_3_utility": agent3_utility_history,
    #     "agent_4_utility": agent4_utility_history
    # }
    #
    # with open(experiments_results_dir, "w") as f:
    #     json.dump(experiments_results, f, indent=4)
