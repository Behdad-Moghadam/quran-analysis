from constants import QURAN_KEYWORDS_FILE_NAME
from main import *

def evaluate(input_text, final_text, method):
    precision_stat, recall_stat, f1_score_stat, bleu_score_stat = total_evaluation(input_text, final_text, method)
    return {
        'max_precision': precision_stat[0],
        'min_precision': precision_stat[1],
        'avg_precision': precision_stat[2],
        'total_precision': precision_stat[3],
        'max_recall': recall_stat[0],
        'min_recall': recall_stat[1],
        'avg_recall': recall_stat[2],
        'total_recall': recall_stat[3],
        'max_f1_score': f1_score_stat[0],
        'min_f1_score': f1_score_stat[1],
        'avg_f1_score': f1_score_stat[2],
        'total_f1_score': f1_score_stat[3],
        'max_bleu_score': bleu_score_stat[0],
        'min_bleu_score': bleu_score_stat[1],
        'avg_bleu_score': bleu_score_stat[2],
        'total_bleu_score': bleu_score_stat[3]
    }

def __find_closest_ayat(keyword, closest_ayat_search_method):
    closest_ayat = None
    if closest_ayat_search_method == "method1":
        closest_ayat = find_closest_ayat_2(keyword)
    elif closest_ayat_search_method == "method2":
        closest_ayat = find_closest_ayat_3(keyword)
    return closest_ayat

def evaluate_main_procedure(keyword, closest_ayat_search_method):
    closest_ayat_list = __find_closest_ayat(keyword, closest_ayat_search_method)
    if len(closest_ayat_list) == 0: return -1
    closest_ayat_text = ''.join(closest_ayat_list)
    preprocessed_en_ayat = ' '.join(preprocess_en_ayat(closest_ayat_text))
    summerized_ayats = summarize_english_ayats(preprocessed_en_ayat)
    final_arabic_text = simple_translate_en2ar(summerized_ayats)
    return evaluate(keyword, final_arabic_text, closest_ayat_search_method)

def evaluate_arabic_summarization(keyword, closest_ayat_search_method):
    closest_ayat = __find_closest_ayat(keyword, closest_ayat_search_method)
    if len(closest_ayat) == 0: return -1
    arabic_closest_ayat_list = postprocess_response(closest_ayat)
    arabic_closest_ayat_text = ''.join(arabic_closest_ayat_list)
    preproc_ayat = preprocess_arabic_ayat(arabic_closest_ayat_text)
    arabic_summary = summarize_arabic_ayat(preproc_ayat)
    return evaluate(keyword, arabic_summary, closest_ayat_search_method)

def evaluate_text_generation(keyword, closest_ayat_search_method):
    closest_ayat_list = __find_closest_ayat(keyword, closest_ayat_search_method)
    if len(closest_ayat_list) == 0: return -1 
    closest_ayat_text = ''.join(closest_ayat_list)
    preproc_ayat_list = preprocess_en_ayat(closest_ayat_text)
    preproc_ayat_text = ''.join(preproc_ayat_list)
    generated_text = generate_english_text(preproc_ayat_text)
    postproc_text = remove_old_text(generated_text, len(preproc_ayat_text))
    postproc_text = remove_not_finished_text(postproc_text)
    final_text = simple_translate_en2ar(postproc_text)
    return evaluate(keyword, final_text, closest_ayat_search_method)

def save_result(evaluation_result: dict|int, count: int, keyword: str, method: str, task: str):
    """
    "total_x" is the x when whole concatenated english ayat is given.
    others are parts of the whole concatenated english ayat result.
    """
    if evaluation_result == -1:
        result = f"{count}|-1\n"
    else:
        max_precision = evaluation_result["max_precision"]
        min_precision = evaluation_result["min_precision"]
        avg_precision = evaluation_result["avg_precision"]
        total_precision = evaluation_result["total_precision"]
        max_recall = evaluation_result["max_recall"]
        min_recall = evaluation_result["min_recall"]
        avg_recall = evaluation_result["avg_recall"]
        total_recall = evaluation_result["total_recall"]
        max_f1_score = evaluation_result["max_f1_score"]
        min_f1_score = evaluation_result["min_f1_score"]
        avg_f1_score = evaluation_result["avg_f1_score"]
        total_f1_score = evaluation_result["total_f1_score"]
        max_bleu_score = evaluation_result["max_bleu_score"]
        min_bleu_score = evaluation_result["min_bleu_score"]
        avg_bleu_score = evaluation_result["avg_bleu_score"]
        total_bleu_score = evaluation_result["total_bleu_score"]
        result = f"{count}|{max_precision}|{min_precision}|{avg_precision}|{total_precision}|{max_recall}|{min_recall}|{avg_recall}|{total_recall}|{max_f1_score}|{min_f1_score}|{avg_f1_score}|{total_f1_score}|{max_bleu_score}|{min_bleu_score}|{avg_bleu_score}|{total_bleu_score}\n"
    save_filename = f"{task}_{method}.txt"
    with open(save_filename, 'a', encoding="utf-8") as f:
        f.write(result)
    print(f"{count}|{keyword}|{task}|{method} Done!")
    

def evaluate_keywords():
    from constants import NOT_IN_QURAN_PERSIAN_WORDS_FILE_NAME
    methods = ["method1", "method2"]
    tasks = ["main", "arabic_summary", "arabic_generation"]
    count = 1
    with open(NOT_IN_QURAN_PERSIAN_WORDS_FILE_NAME, 'r', encoding='utf-8') as f:
        keyword = f.readline()
        while keyword:
            if count <= 62:  # to 80                 # Manual Resume Mechanism
                count += 1
                keyword = f.readline()
                continue
            elif count == 81: break
            keyword = keyword.strip()
            for method in methods:
                for task in tasks:
                    evaluation_function = None
                    if task == "main":
                        evaluation_function = evaluate_main_procedure
                    elif task == "arabic_summary":
                        evaluation_function = evaluate_arabic_summarization
                    elif task == "arabic_generation":
                        evaluation_function = evaluate_text_generation
                    evaluation_result = evaluation_function(keyword, method)
                    if evaluation_result == -1 and task == "main":
                        save_result(evaluation_result, count, keyword, method, "main")
                        save_result(evaluation_result, count, keyword, method, "arabic_summary")
                        save_result(evaluation_result, count, keyword, method, "arabic_generation")
                        break
                    else:
                        save_result(evaluation_result, count, keyword, method, task)
            count += 1
            keyword = f.readline()
        
def postprocess_keywords():
    from constants import KEYWORD_NAMES_FILE_NAME
    postproc_keywords = list()
    with open(KEYWORD_NAMES_FILE_NAME, 'r', encoding="utf-8") as f:
        keyword = f.readline()
        while keyword:
            postproc_keyword = keyword.replace('ي', 'ی')
            postproc_keyword = postproc_keyword.replace('ى', 'ی')     # replace 'ىا' with 'یا'
            postproc_keyword = postproc_keyword.replace('ك', 'ک')
            postproc_keywords.append(postproc_keyword)
            keyword = f.readline()
    with open("postproc_quran_names_keywords.txt", 'a', encoding="utf-8") as f:
        for keyword in postproc_keywords:
            f.write(keyword)

if __name__ == "__main__":
    evaluate_keywords()

    # count = 2263
    # keyword = "قدرتمند"
    # task = "arabic_generation"
    # method = "method2"
    # if task == "main":
    #     evaluation_function = evaluate_main_procedure
    # elif task == "arabic_summary":
    #     evaluation_function = evaluate_arabic_summarization
    # elif task == "arabic_generation":
    #     evaluation_function = evaluate_text_generation
    # evaluation_result = evaluation_function(keyword, method)
    # if evaluation_result == -1 and task == "main":
    #     save_result(evaluation_result, count, keyword, method, "main")
    #     save_result(evaluation_result, count, keyword, method, "arabic_summary")
    #     save_result(evaluation_result, count, keyword, method, "arabic_generation")
    # else:
    #     save_result(evaluation_result, count, keyword, method, task)