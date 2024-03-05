def _check_error(data_file_name: str):
    line_counter = 1
    counter_program = 1
    with open(data_file_name, 'r', encoding="utf-8") as f:
        line = f.readline()
        while line:
            counter_in_file = int(line.split('|')[0])
            if counter_in_file == counter_program:
                counter_program += 1
            elif counter_in_file < line_counter:
                print(f"at line {line_counter}, found lower counter.")
                counter_program = counter_in_file + 1
            elif counter_in_file > line_counter:
                print(f"at line {line_counter}, found higher counter.")
                counter_program = counter_in_file + 1
            line = f.readline()
            line_counter += 1

def _check_higher_than_one_values(src_file_path: str):
    with open(src_file_path, 'r', encoding='utf-8') as src_file:
        src_line = src_file.readline()
        while src_line:
            src_sections = list(map(lambda x: float(x.strip()), src_line.split('|')))
            for idx, section in enumerate(src_sections):
                if idx == 0: continue
                if section > 1.0:
                    print("line:", src_sections[0], "value:", section)
            src_line = src_file.readline()


def check_files():
    from constants import NAMES_IN_QURAN_EVAL_RESULTS_DIR, VERBS_IN_QURAN_EVAL_RESULTS_DIR,\
                        NOT_IN_QURAN_EVAL_RESULTS_DIR, ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR,\
                        EVALUATION_DATA_MAIN_METHOD1_FILE_NAME, EVALUATION_DATA_MAIN_METHOD2_FILE_NAME,\
                        EVALUATION_DATA_SUM_METHOD1_FILE_NAME, EVALUATION_DATA_SUM_METHOD2_FILE_NAME,\
                        EVALUATION_DATA_GEN_METHOD1_FILE_NAME, EVALUATION_DATA_GEN_METHOD2_FILE_NAME
    dir_names = [NAMES_IN_QURAN_EVAL_RESULTS_DIR, VERBS_IN_QURAN_EVAL_RESULTS_DIR, ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR, NOT_IN_QURAN_EVAL_RESULTS_DIR]
    file_names = [EVALUATION_DATA_MAIN_METHOD1_FILE_NAME, EVALUATION_DATA_MAIN_METHOD2_FILE_NAME,
                    EVALUATION_DATA_SUM_METHOD1_FILE_NAME, EVALUATION_DATA_SUM_METHOD2_FILE_NAME,
                    EVALUATION_DATA_GEN_METHOD1_FILE_NAME, EVALUATION_DATA_GEN_METHOD2_FILE_NAME]
    print("test")
    for dir_name in dir_names:
        for file_name in file_names:
            file_path = dir_name + "/" + file_name
            dst_file_path = dir_name + "/fixed/" + file_name 
            print(dst_file_path+":")
            _check_higher_than_one_values(dst_file_path)
            print("*"*10)


def get_data(data_file_name: str):
    data = dict(max_precisions=[],min_precisions=[],avg_precisions=[],total_precisions=[],
                max_recalls=[], min_recalls=[], avg_recalls=[], total_recalls=[],
                max_f1_scores=[], min_f1_scores=[], avg_f1_scores=[], total_f1_scores=[],
                max_bleu_scores=[], min_bleu_scores=[], avg_bleu_scores=[], total_bleu_scores=[],
                not_found_words=[])
    with open(data_file_name, 'r', encoding="utf-8") as f:
        line = f.readline()
        while line:
            sections = [float(section.strip()) for section in line.split('|')]
            counter = int(sections[0])
            if sections[1] == -1:
                data["not_found_words"].append(counter)
                line = f.readline()
                continue
            else:
                data["max_precisions"].append(sections[1])    
                data["min_precisions"].append(sections[2])
                data["avg_precisions"].append(sections[3])
                data["total_precisions"].append(sections[4])
                data["max_recalls"].append(sections[5])
                data["min_recalls"].append(sections[6])
                data["avg_recalls"].append(sections[7])
                data["total_recalls"].append(sections[8])
                data["max_f1_scores"].append(sections[9])
                data["min_f1_scores"].append(sections[10])
                data["avg_f1_scores"].append(sections[11])
                data["total_f1_scores"].append(sections[12])
                data["max_bleu_scores"].append(sections[13])
                data["min_bleu_scores"].append(sections[14])
                data["avg_bleu_scores"].append(sections[15])
                data["total_bleu_scores"].append(sections[16])
                line = f.readline()
    return data
    
def get_number_of_common_words(not_found_words_method1, not_found_words_method2):
    not_found_words_method1_set = set(not_found_words_method1)
    not_found_words_method2_set = set(not_found_words_method2)
    return len(not_found_words_method1_set.intersection(not_found_words_method2_set))
    
def __init_all_factors_data():
    return dict(max_precisions=None, min_precisions=None,
                avg_precisions=None, total_precisions=None,
                max_recalls=None, min_recalls=None,
                avg_recalls=None, total_recalls=None,
                max_f1_scores=None, min_f1_scores=None,
                avg_f1_scores=None, total_f1_scores=None,
                max_bleu_scores=None, min_bleu_scores=None,
                avg_bleu_scores=None, total_bleu_scores=None)

def __calculate_max_min_sum_value(cur_value, max_value, min_value, sum_value):
    if cur_value > max_value:
        max_value = cur_value
    if cur_value < min_value:
        min_value = cur_value
    sum_value += cur_value
    return max_value, min_value, sum_value

def __get_statistics_for_one_factor(method1_factor_data, method2_factor_data):
    from statistics import mean
    better_value1, better_value2 = 0,0
    for value1, value2 in zip(method1_factor_data, method2_factor_data):
        if value1 > value2:
            better_value1 += 1
        elif value2 > value1:
            better_value2 += 1
    max_value1, max_value2 = max(method1_factor_data), max(method2_factor_data)
    min_value1, min_value2 = min(method1_factor_data), min(method2_factor_data)
    avg_value1, avg_value2 = mean(method1_factor_data), mean(method2_factor_data)
    return dict(method1=dict(max=max_value1, min=min_value1, avg=avg_value1, better_value_count=better_value1),
                method2=dict(max=max_value2, min=min_value2, avg=avg_value2, better_value_count=better_value2))

def _get_statistics_for_one_task(method1_data, method2_data):
    from copy import deepcopy
    method1_statistics = __init_all_factors_data()
    method2_statistics = __init_all_factors_data()
    method1_to_method2_comparison = __init_all_factors_data()
    cur_task_statistics = dict(method1=dict(), method2=dict())
    for method_factor in method1_data:
        if method_factor == "not_found_words": continue
        factor_statistics = __get_statistics_for_one_factor(method1_data[method_factor], method2_data[method_factor])
        cur_task_statistics["method1"][method_factor] = factor_statistics["method1"]
        cur_task_statistics["method2"][method_factor] = factor_statistics["method2"]
    return cur_task_statistics

def get_total_statistics(main_method1_data, main_method2_data,
                    summerization_method1_data, summerization_method2_data,
                    generation_method1_data, generation_method2_data):
    statistics = dict(number_of_not_found_words_method1=-1, number_of_not_found_words_method2=-1,
                        number_of_common_not_found_words=-1, main_task=dict(), summarization_task=dict(),
                        generation_task=dict())
    # Finding the not found words of one task is enough, ex. main task 
    statistics["number_of_not_found_words_method1"] = len(main_method1_data["not_found_words"])
    statistics["number_of_not_found_words_method2"] = len(main_method2_data["not_found_words"])
    statistics["number_of_common_not_found_words"] = get_number_of_common_words(main_method1_data["not_found_words"],
                                                                    main_method2_data["not_found_words"])
    statistics["main_task"] = _get_statistics_for_one_task(main_method1_data, main_method2_data)
    statistics["summarization_task"] = _get_statistics_for_one_task(summerization_method1_data, summerization_method2_data)
    statistics["generation_task"] = _get_statistics_for_one_task(generation_method1_data, generation_method2_data)
    return statistics

def make_stacked_bars_charts(main_method1_data, main_method2_data,
                summarization_method1_data, summarization_method2_data,
                generation_method1_data, generation_method2_data):
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    import numpy as np
    datasets = [main_method1_data, main_method2_data,
                summarization_method1_data, summarization_method2_data,
                generation_method1_data, generation_method2_data]
    bins = np.arange(min(list(map(min,datasets))), max(list(map(max,datasets))) + 0.05, 0.05)
    colors = matplotlib.cm.tab20(range(6))
    methods = ["Main Method1", "Main Method2", "Summarization Method1", "Summarization Method2", "Generation Method1", "Generation Method2"]
    plt.hist(datasets, bins = bins, color = colors, label=methods)
    plt.xlabel('Min Bleu Scores')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def make_density_plots(main_method1_data, main_method2_data,
                summarization_method1_data, summarization_method2_data,
                generation_method1_data, generation_method2_data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    datasets = [main_method1_data, main_method2_data,
                summarization_method1_data, summarization_method2_data,
                generation_method1_data, generation_method2_data]
    methods = ["Main Method1", "Main Method2", "Summarization Method1", "Summarization Method2", "Generation Method1", "Generation Method2"]
    for dataset, method in zip(datasets,methods):
        sns.distplot(dataset, hist = False,
                    # kde = True,
                    # kde_kws = {'linewidth': 3},
                    label = method)
    plt.xlabel('Min Bleu Scores')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def _get_file_lines_count(filename):
    with open(filename, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines

def join2files(filename1, filename2):
    next_line_number = _get_file_lines_count(filename1) + 1
    with open(filename1, 'a', encoding="utf-8") as f1:
        with open(filename2, 'r', encoding="utf-8") as f2:
            read_line = f2.readline()
            while read_line:
                splits = read_line.split('|')
                splits[0] = str(next_line_number)
                new_line = '|'.join(splits)
                f1.write(new_line)
                read_line = f2.readline()
                next_line_number += 1
    
def join_all_files():
    from constants import EVALUATION_DATA_MAIN_METHOD1_FILE_NAME, EVALUATION_DATA_MAIN_METHOD2_FILE_NAME,\
                    EVALUATION_DATA_SUM_METHOD1_FILE_NAME, EVALUATION_DATA_SUM_METHOD2_FILE_NAME,\
                    EVALUATION_DATA_GEN_METHOD1_FILE_NAME, EVALUATION_DATA_GEN_METHOD2_FILE_NAME,\
                    ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR

    file_names = [EVALUATION_DATA_MAIN_METHOD1_FILE_NAME, EVALUATION_DATA_MAIN_METHOD2_FILE_NAME,
                    EVALUATION_DATA_SUM_METHOD1_FILE_NAME, EVALUATION_DATA_SUM_METHOD2_FILE_NAME,
                    EVALUATION_DATA_GEN_METHOD1_FILE_NAME, EVALUATION_DATA_GEN_METHOD2_FILE_NAME]
    dst_dir = ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR
    src_dir = ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR + "/verbs"
    for file_name in file_names:
        filename1 = dst_dir + "/" + file_name
        filename2 = src_dir + "/" + file_name   
        join2files(filename1, filename2)

def main():
    from constants import EVALUATION_DATA_MAIN_METHOD1_FILE_NAME, EVALUATION_DATA_MAIN_METHOD2_FILE_NAME, EVALUATION_DATA_SUM_METHOD1_FILE_NAME, EVALUATION_DATA_SUM_METHOD2_FILE_NAME, EVALUATION_DATA_GEN_METHOD1_FILE_NAME, EVALUATION_DATA_GEN_METHOD2_FILE_NAME,\
        ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR
    import json
    evaluation_main_method1_data = ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR + "/fixed/" + EVALUATION_DATA_MAIN_METHOD1_FILE_NAME
    evaluation_main_method2_data = ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR + "/fixed/" + EVALUATION_DATA_MAIN_METHOD2_FILE_NAME
    evaluation_sum_method1_data = ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR + "/fixed/" + EVALUATION_DATA_SUM_METHOD1_FILE_NAME
    evaluation_sum_method2_data = ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR + "/fixed/" + EVALUATION_DATA_SUM_METHOD2_FILE_NAME
    evaluation_gen_method1_data = ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR + "/fixed/" + EVALUATION_DATA_GEN_METHOD1_FILE_NAME
    evaluation_gen_method2_data = ALL_WORDS_IN_QURAN_EVAL_RESULTS_DIR + "/fixed/" + EVALUATION_DATA_GEN_METHOD2_FILE_NAME
    main_method1_data = get_data(evaluation_main_method1_data)
    main_method2_data = get_data(evaluation_main_method2_data)
    sum_method1_data = get_data(evaluation_sum_method1_data)
    sum_method2_data = get_data(evaluation_sum_method2_data)
    gen_method1_data = get_data(evaluation_gen_method1_data)
    gen_method2_data = get_data(evaluation_gen_method2_data)
    statistics = get_total_statistics(main_method1_data, main_method2_data,
                                    sum_method1_data, sum_method2_data,
                                    gen_method1_data, gen_method2_data)
    print(json.dumps(statistics, indent=4))

    factor = "min_bleu_scores"

    make_density_plots(main_method1_data[factor], main_method2_data[factor],
                sum_method1_data[factor], sum_method2_data[factor],
                gen_method1_data[factor], gen_method2_data[factor])
    # Make Chart for Max Precisions
    make_stacked_bars_charts(main_method1_data[factor], main_method2_data[factor],
                sum_method1_data[factor], sum_method2_data[factor],
                gen_method1_data[factor], gen_method2_data[factor])
    


if __name__ == "__main__":
    # check_files()             # in order to check evalutaion results correctness, uncomment this line
    main()
    