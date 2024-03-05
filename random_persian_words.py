from constants import ALL_PERSIAN_WORDS_FILE_NAME, NOT_IN_QURAN_PERSIAN_WORDS_FILE_NAME, \
                        KEYWORD_NAMES_FILE_NAME, KEYWORD_VERBS_FILE_NAME
from copy import copy

def make_n_random_numbers_between_two_numbers(n, number1, number2, exclude_list=[]):
    import random
    if number1 >= number2:
        raise ValueError("number1 should be less than number2")

    exclude_set = set(exclude_list)

    result = []
    while len(result) < n:
        random_number = random.randint(number1, number2)
        if random_number not in exclude_set and random_number not in result:
            result.append(random_number)

    return result

def write_in_file(file_name, items: list):
    with open(file_name, 'a', encoding="utf-8") as f:
        for item in items:
            f.write(item.strip()+"\n")

def make_list_of_words_from_file(filename):
    words = list()
    with open(filename, 'r', encoding="utf-8") as f:
        words.append(f.readline())
    return words

def make_n_random_persian_words(n, exclude_list=[]):
    random_numbers = make_n_random_numbers_between_two_numbers(n, 1, 453150, exclude_list)
    sorted_random_numbers = sorted(random_numbers)
    random_persian_words = list()
    with open(ALL_PERSIAN_WORDS_FILE_NAME,'r', encoding="utf-8") as f:
        line_count = 1
        occurs_count = 0
        cur_word = f.readline()
        while(occurs_count<n):                
            if line_count == sorted_random_numbers[occurs_count]:
                occurs_count += 1
                random_persian_words.append(cur_word)
            cur_word = f.readline()
            line_count += 1
    return random_persian_words, random_numbers

if __name__ == "__main__":
    TOTAL_WORDS_WANTED = 3000
    quran_names_list = make_list_of_words_from_file(KEYWORD_NAMES_FILE_NAME)
    quran_verbs_list = make_list_of_words_from_file(KEYWORD_VERBS_FILE_NAME)
    quran_persian_words = copy(quran_names_list)
    quran_persian_words.extend(quran_verbs_list)
    print(quran_persian_words)
    random_persian_words, random_numbers = make_n_random_persian_words(TOTAL_WORDS_WANTED)
    persian_words_not_in_quran = set(random_persian_words) - set(quran_persian_words)
    new_total_words_wanted = TOTAL_WORDS_WANTED - len(persian_words_not_in_quran)
    while new_total_words_wanted > 0:
        random_persian_words, new_random_numbers = make_n_random_persian_words(new_total_words_wanted, random_numbers)
        random_numbers.extend(new_random_numbers)
        new_persian_words_not_in_quran = random_persian_words - quran_persian_words
        persian_words_not_in_quran.update(new_persian_words_not_in_quran)
        new_total_words_wanted = TOTAL_WORDS_WANTED - len(persian_words_not_in_quran)
    write_in_file(NOT_IN_QURAN_PERSIAN_WORDS_FILE_NAME, list(persian_words_not_in_quran))