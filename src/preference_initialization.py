import json

import numpy as np
from numpy.linalg import eig
from numpy import unravel_index


def calculate_membership(value, integer_points):
    """
    the function for calculating the values of membership functions of agent's preference
    :param value: the value from the fuzzy likert scale questionnaire
    :param integer_points: the pre-defined integer values of the attitudes. For now, the predefined values are integers
    from -2 to 2, representing the attitudes (utilities) from "strongly disagree" to "strongly agree".
    :return: membership function values
    """
    # Membership values for each integer point
    memberships = {point: max(round(1 - abs(value - point), 2), 0) for point in integer_points}
    print("memberships*", memberships)
    return memberships


def calculate_trapezoid_area(base1, base2, height):
    """
    calculation of the defuzzification of membership functions, which is a trapezoid area
    :param base1: the lower edge of the trapezoid area
    :param base2: the higher edge of the trapezoid area
    :param height: the height of the trapezoid area
    :return:
    """
    # Calculate the area of a trapezoid
    return 0.5 * (base1 + base2) * height


def defuzzification_fuzzy_likert(response_memberships):
    """
    generalized defuzzification function.
    :param response_memberships: the response results of the questionnaire
    :return:
    """
    total_area = 0
    weighted_sum = 0

    for level, membership in response_memberships.items():
        if membership > 0:
            # Calculate the area of each membership function (assumed to be trapezoidal)
            fixed_base = 1.0  # Distance between the integer points on the scale (for example, 4 to 5)
            variable_base = (1 - membership) * fixed_base  # The same for simplicity; this can be adjusted if needed
            height = membership
            area = calculate_trapezoid_area(fixed_base, variable_base, height)

            # Calculate weighted sum and total area for defuzzification
            total_area += area
            weighted_sum += level * area

    if total_area == 0:
        return None  # Avoid division by zero

    return weighted_sum / total_area


# fixed points
integer_points = [-2, -1, 0, 1, 2]


def measure_preference_from_answers(integer_points: list[int]):
    answer_dir = "./questionnaire/answers_test.json"
    with open(answer_dir, "r") as f:
        questionnaire_answers = json.load(f)

    questionnaire_dir = "./questionnaire/questionnaire_test.json"
    with open(questionnaire_dir, "r") as f:
        opinions_list = json.load(f)

    defuzzification_value_list = []
    for answer in questionnaire_answers:
        answer_membership = calculate_membership(value=answer["answer_value"], integer_points=integer_points)
        defuzzification_value = defuzzification_fuzzy_likert(response_memberships=answer_membership)
        defuzzification_value_list.append(defuzzification_value)

    opinions_and_attitudes = []
    for i in range(0, len(opinions_list)):
        opinion_and_attitude = {
            "opinion": opinions_list[i]["text"],
            "attitude": round(defuzzification_value_list[i], 2)
        }
        opinions_and_attitudes.append(opinion_and_attitude)

    return opinions_and_attitudes


opinions_and_attitudes_test = measure_preference_from_answers(integer_points)
print(opinions_and_attitudes_test)


def defuzzification_alpha_cut(importance_level: int):
    # initialize triangle fuzzy number
    if importance_level == 1:
        l = 1
        m = 1
        u = 2
    elif importance_level == 2:
        l = 2
        m = 3
        u = 4
    elif importance_level == 3:
        l = 4
        m = 5
        u = 6
    elif importance_level == 4:
        l = 6
        m = 7
        u = 8
    elif importance_level == 5:
        l = 8
        m = 9
        u = 10

    # initialize alpha
    alpha = 0.5

    # defuzzify triangle fuzzy numbers to alpha intervals
    l_alpha = l + alpha * (m - l)
    u_alpha = u - alpha * (u - m)

    # defuzzify alpha intervals to a simple number
    defuzzified_alpha = (l_alpha + u_alpha) / 2
    print("defuzzified_alpha: ", defuzzified_alpha)

    return defuzzified_alpha


def matrix_normalization(matrix: np.ndarray):
    column_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / column_sums

    return normalized_matrix


def calculate_left_eigenvector(matrix: np.array):
    matrix_transpose = matrix.T
    eigenvalue, eigenvector = eig(matrix_transpose)
    max_index = np.argmax(eigenvalue)
    left_eigenvector = eigenvector[:, max_index].real
    left_eigenvector_normalized = left_eigenvector / sum(left_eigenvector)
    return left_eigenvector_normalized


def calculate_principle_eigenvalue_and_eigenvector(matrix: np.array):
    adjusted_matrix = matrix
    eigenvalue, eigenvector = eig(adjusted_matrix)
    max_index = np.argmax(eigenvalue)
    principle_eigenvalue = eigenvalue[max_index].real
    principle_eigenvector = eigenvector[:, max_index].real
    principle_eigenvector_normalized = principle_eigenvector / sum(principle_eigenvector)
    return principle_eigenvalue, principle_eigenvector_normalized


def calculate_partial_derivatives(matrix, principle_eigenvector, left_eigenvector, n):
    partial_derivatives_matrix = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            partial_derivatives_matrix[i, j] = ((left_eigenvector[i] * principle_eigenvector[j]) -
                                                (1 / matrix[i, j] ** 2) *
                                                (left_eigenvector[j] * principle_eigenvector[i]))

    return partial_derivatives_matrix


def calculate_consistency_ratio(principle_eigenvalue, n, random_index):
    consistency_index = (principle_eigenvalue - n) / (n - 1)
    consistency_ratio = consistency_index / random_index

    return consistency_ratio


def adjust_matrix_to_near_consistency(matrix: np.array):
    adjusted_matrix = matrix
    n = adjusted_matrix.shape[0]
    principle_eigenvalue, principle_eigenvector = calculate_principle_eigenvalue_and_eigenvector(adjusted_matrix)
    random_index = 0.58
    consistency_ratio = calculate_consistency_ratio(principle_eigenvalue=principle_eigenvalue,
                                                    n=n, random_index=random_index)
    print("consistency_ratio:", consistency_ratio)
    if round(consistency_ratio, 2) > 0.1:
        left_eigenvector = calculate_left_eigenvector(matrix=adjusted_matrix)
        partial_derivatives_matrix = calculate_partial_derivatives(matrix=adjusted_matrix,
                                                                   principle_eigenvector=principle_eigenvector,
                                                                   left_eigenvector=left_eigenvector, n=n)
        max_indexes = unravel_index(partial_derivatives_matrix.argmax(), partial_derivatives_matrix.shape)
        # plus adjustment
        adjusted_matrix[max_indexes[0], max_indexes[1]] = (adjusted_matrix[max_indexes[0], max_indexes[1]] +
                                                           0.001)
        adjusted_matrix[max_indexes[1], max_indexes[0]] = np.reciprocal(
            adjusted_matrix[max_indexes[0], max_indexes[1]])
        principle_eigenvalue_plus, principle_eigenvector_plus = calculate_principle_eigenvalue_and_eigenvector(
            adjusted_matrix)
        consistency_ratio_plus = calculate_consistency_ratio(principle_eigenvalue=principle_eigenvalue_plus,
                                                             n=n, random_index=random_index)
        if consistency_ratio_plus > consistency_ratio:
            print("option 1")
            adjusted_matrix[max_indexes[0], max_indexes[1]] = (adjusted_matrix[max_indexes[0], max_indexes[1]] -
                                                               0.002)
            adjusted_matrix[max_indexes[1], max_indexes[0]] = np.reciprocal(
                adjusted_matrix[max_indexes[0], max_indexes[1]])
            principle_eigenvector = adjust_matrix_to_near_consistency(matrix=adjusted_matrix)
            return principle_eigenvector
        elif consistency_ratio_plus < consistency_ratio:
            print("option 2")
            principle_eigenvector = adjust_matrix_to_near_consistency(matrix=adjusted_matrix)
            return principle_eigenvector
        else:
            return principle_eigenvector
    else:
        return principle_eigenvector


def get_priority_vector():
    alpha = 0.5

    comparison_results_dir = "./questionnaire/comparison_test.json"
    with open(comparison_results_dir, "r") as f:
        comparison_results_list = json.load(f)

    fuzzy_ahp_array = np.empty(shape=(len(comparison_results_list), len(comparison_results_list)))
    fuzzy_ahp_array.fill(1)

    # matrix defuzzification
    for comparison_result in comparison_results_list:
        x_index = comparison_result["selected_question_id"] - 1
        y_index = comparison_result["base_question_id"] - 1
        defuzzification_ahp_array_value = defuzzification_alpha_cut(
            importance_level=comparison_result["importance_level"])
        fuzzy_ahp_array[x_index, y_index] = defuzzification_ahp_array_value
        fuzzy_ahp_array[y_index, x_index] = np.reciprocal(defuzzification_ahp_array_value)

    # # normalized numpy matrix
    # normalized_fuzzy_ahp_array = matrix_normalization(fuzzy_ahp_array)
    #
    # # calculate priority vector
    # priority_vector_np = normalized_fuzzy_ahp_array.mean(axis=1)
    # priority_vector_list = priority_vector_np.tolist()
    # priority_vector_list = [round(priority_vector_list[i], 3) for i in range(len(priority_vector_list))]

    # Transfer the matrix to near consistency
    print("original numpy matrix\n", fuzzy_ahp_array)
    principle_eigenvector = adjust_matrix_to_near_consistency(matrix=fuzzy_ahp_array)
    principle_eigenvector_list = principle_eigenvector.tolist()
    for i in range(0, len(principle_eigenvector_list)):
        principle_eigenvector_list[i] = round(principle_eigenvector_list[i], 2)

    return principle_eigenvector_list


priority_vector_test = get_priority_vector()
print("priority vector: ", priority_vector_test)


def combine_opinions_attitudes_and_importance(opinions_and_attitudes: list, priority_list: list):
    opinions_and_attitudes_dir = "./questionnaire/combination_test.json"

    for i in range(0, len(opinions_and_attitudes)):
        opinions_and_attitudes[i]["importance"] = priority_list[i]
        opinions_and_attitudes[i]["index"] = i+1
    with open(opinions_and_attitudes_dir, "w") as f:
        json.dump(opinions_and_attitudes, f, indent=4)

    return opinions_and_attitudes


combine_opinions_attitudes_and_importance(opinions_and_attitudes=opinions_and_attitudes_test,
                                          priority_list=priority_vector_test)
