__author__ = "Dong Yihan"

import json


def convert_utility_to_language(combination_dir):
    """
    convert numeric utilities to vague language descriptions for pure LLM-based agents.
    :param combination_dir: combination directory
    :return:
    """

    with open(combination_dir, "r") as f1:
        agent_preference = json.load(f1)
    agent_utility_list = [element["attitude"] for element in agent_preference]

    base_labels = [
        (-2.0, "Strongly Disagree"),
        (-1.2, "Disagree"),
        (-0.6, "Slightly Disagree"),
        (0.0, "Neutral"),
        (0.6, "Slightly Agree"),
        (1.2, "Agree"),
        (2.0, "Strongly Agree")
    ]

    certainty_edges = {
        "probably": 0.3,
        "somewhat": 0.2,
        "leaning": 0.1
    }

    language_descriptions = []
    for utility in agent_utility_list:
        for i in range(len(base_labels) - 1):
            lower, label_a = base_labels[i]
            upper, label_b = base_labels[i + 1]
            if lower <= utility <= upper:
                midpoint = (lower + upper) / 2
                delta = abs(utility - midpoint)
                if delta <= certainty_edges["leaning"]:
                    language_descriptions.append(f"Slightly leaning towards "
                                                 f"{label_b if utility > midpoint else label_a}")
                elif delta <= certainty_edges["somewhat"]:
                    language_descriptions.append(f"Somewhat between {label_a} and {label_b}")
                elif delta <= certainty_edges["probably"]:
                    language_descriptions.append(f"Probably {label_b if utility > midpoint else label_a}")
                else:
                    language_descriptions.append(f"Definitely {label_b if utility > midpoint else label_a}")

    for i in range(0, len(agent_preference)):
        agent_preference[i]["attitude_description"] = language_descriptions[i]

    with open(combination_dir, "w") as f2:
        json.dump(agent_preference, f2, indent=4)

    return


def convert_significance_weights_to_language(combination_dir):
    """
    convert numeric significance weights to language
    :param combination_dir:
    :return:
    """
    with open(combination_dir, "r") as f1:
        agent_preference = json.load(f1)
    agent_significance_list = [element["importance"] for element in agent_preference]

    language_descriptions = []
    for significance_weight in agent_significance_list:
        if significance_weight >= 0.8:
            language_descriptions.append(f"Extremely important")
        elif significance_weight >= 0.6:
            language_descriptions.append(f"Very important")
        elif significance_weight >= 0.4:
            language_descriptions.append(f"Moderately important")
        elif significance_weight >= 0.2:
            language_descriptions.append(f"Slightly important")
        else:
            language_descriptions.append(f"Not important at all")

    for i in range(0, len(agent_preference)):
        agent_preference[i]["significance_description"] = language_descriptions[i]

    with open(combination_dir, "w") as f2:
        json.dump(agent_preference, f2, indent=4)

    return


def convert_language_to_utility(combination_dir):
    """
    The reversing process of convertion of utility to language. Use this function after the experiment to check
     the results of discussions
    :param combination_dir:
    :return:
    """

    with open(combination_dir, "r") as f1:
        agent_attitude_data = json.load(f1)
    agent_attitude_list = [element["attitude description"] for element in agent_attitude_data]

    # Base labels and their reference crisp values (center points)
    base_to_value = {
        "Strongly Disagree": -2.0,
        "Disagree": -1.2,
        "Slightly Disagree": -0.6,
        "Neutral": 0.0,
        "Slightly Agree": 0.6,
        "Agree": 1.2,
        "Strongly Agree": 2.0
    }

    # Modifier offsets from the base value (tunable)
    modifier_offsets = {
        "Definitely": 0.1,
        "Probably": 0.2,
        "Somewhat between": 0.3,  # used with two bases, will take midpoint
        "Slightly leaning towards": 0.4
    }

    attitude_defuzzify_value = []

    for attitude_description in agent_attitude_list:
        if attitude_description.startwith("Somewhat between"):
            stripped = attitude_description.replace("Somewhat between", "").strip()
            if " and " in stripped:
                base1, base2 = stripped.split(" and ")
                val1 = base_to_value[base1.strip()]
                val2 = base_to_value[base2.strip()]
                return round((val1 + val2) / 2, 2)

        for modifier, offset in modifier_offsets.items():
            if attitude_description.startwith(modifier):
                base_phase = attitude_description.replace(modifier, "").strip()
                base_value = base_to_value.get(base_phase)
                if base_value is not None:
                    direction = 1 if base_value < 0 else -1 if base_value > 0 else 0
                    attitude_defuzzify_value.append(round(base_value + direction * offset, 2))


if __name__ == "__main__":
    agent_combination_dir = "./combination/agent2.json"
    convert_utility_to_language(combination_dir=agent_combination_dir)
    convert_significance_weights_to_language(combination_dir=agent_combination_dir)
