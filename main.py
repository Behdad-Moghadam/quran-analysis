from copy import copy
from database import add_farsi_arabic_text, add_farsi_english_text, Method, get_arabic_texts, get_english_texts

########################## Find Closest Ayat ##########################
def find_closest_ayat(text):
    import requests
    API_URL = 'https://hadith.ai/search/'
    data = ('{"query": "'+text+'", "method": "sent"}').encode('utf-8')
    raw_api_response = requests.post(API_URL, data).json()["output"]
    ayat_should_be_passed, total_acceptable_ayat_count = find_ayat_should_be_passed(raw_api_response)
    ayat_text = find_ayat_text_from_file(ayat_should_be_passed, total_acceptable_ayat_count, text)
    return ayat_text

def find_closest_ayat_2(farsi_text):
    import requests
    from bs4 import BeautifulSoup
    arabic_texts = get_arabic_texts(farsi_text, method=Method.ONE)
    english_texts = get_english_texts(farsi_text, method=Method.ONE)
    if len(arabic_texts) != 0 and len(english_texts) != 0:
        return __concat_arabic_and_english_texts(arabic_texts, english_texts)
    API_URL = "https://ayat.language.ml/lib/searchQ.php"
    params = {"q": farsi_text, "method": "sent"}
    raw_api_response = requests.get(API_URL, params=params)
    response_content = raw_api_response.text
    soup = BeautifulSoup(response_content, 'html.parser')
    results = [div.text for div in soup.findAll('div', class_="result_head")]
    surah_names_with_ayeh_numbers_and_precision = [extract_surah_and_ayeh(result) for result in results]
    ayat_should_be_passed, total_acceptable_ayat_count = find_ayat_should_be_passed_2nd_api(surah_names_with_ayeh_numbers_and_precision)
    ayat_text = find_ayat_text_from_file(ayat_should_be_passed, total_acceptable_ayat_count, farsi_text)
    return ayat_text

arabic_ayat = dict()    # (surah_no, ayeh_no): arabic_ayeh
english_ayat = dict()   # (surah_no, ayeh_no): english_ayeh

def find_closest_ayat_3(farsi_text) -> list:
    arabic_texts = get_arabic_texts(farsi_text, method=Method.TWO)
    english_texts = get_english_texts(farsi_text, method=Method.TWO)
    if len(arabic_texts) != 0 and len(english_texts) != 0:
        return __concat_arabic_and_english_texts(arabic_texts, english_texts)
    english_text = translate_fa2en_google(farsi_text)
    closest_similarity_ayat = find_closest_similarity_ayat_2(english_text)
    if closest_similarity_ayat is None:
        return []
    final_ayat_text = list()
    all_preprocessed_arabic_texts = ""
    all_english_texts = ""
    for key, similarity in closest_similarity_ayat.items():
        preprocess_arabic_ayeh = preprocess_ar_ayeh(arabic_ayat[key].strip())
        all_preprocessed_arabic_texts = all_preprocessed_arabic_texts + "#" + preprocess_arabic_ayeh
        current_english_text = english_ayat[key].strip() + f" ({key})"
        all_english_texts = all_english_texts + "#" + current_english_text
        final_ayat_text.append(preprocess_arabic_ayeh + " => " + f"{current_english_text}\n")
    add_farsi_arabic_text(farsi_text, all_preprocessed_arabic_texts, method=Method.TWO)
    add_farsi_english_text(farsi_text, all_english_texts, method=Method.TWO)
    return final_ayat_text

def __concat_arabic_and_english_texts(arabic_texts, english_texts):
    source_arabic_text = arabic_texts[0][0]
    source_english_text = english_texts[0][0]
    arabic_ayat = source_arabic_text.split("#")
    english_ayat = source_english_text.split("#")
    final_ayat_text = list()
    for arabic_ayeh, english_ayeh in zip(arabic_ayat, english_ayat):
        if len(arabic_ayeh) == 0 and len(english_ayeh) == 0: continue
        final_ayat_text.append(f"{arabic_ayeh} => {english_ayeh}\n")
    return final_ayat_text

def find_closest_similarity_ayat(english_text):
    from sentence_similarity import sentence_similarity         # This line doesn't work!
    model = sentence_similarity(model_name="sentence-transformers/all-MiniLM-L6-v2", embedding_type="cls_token_embedding")
    similarity_ayat = dict()
    init_arabic_and_english_ayat_dict()
    for key, en_ayeh in english_ayat.items():
        score = model.get_score(english_text, en_ayeh, metric="cosine")
        similarity_ayat[key] = score
    similarity_ayat = dict(sorted(similarity_ayat.items(), key=lambda item: item[1], reverse=True))
    closest_similarity_ayat = dict()
    for idx, key in enumerate(similarity_ayat):
        similarity = similarity_ayat[key]
        if idx == 10 or similarity < 0.4:
            break
        closest_similarity_ayat[key] = similarity
    return closest_similarity_ayat if len(closest_similarity_ayat) != 0 else None
    
def find_closest_similarity_ayat_2(english_text):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    english_text_emb = model.encode(english_text, convert_to_tensor=True)
    similarity_ayat = dict()
    init_arabic_and_english_ayat_dict()
    for key, en_ayeh in english_ayat.items():
        en_ayeh_embedding = model.encode(en_ayeh, convert_to_tensor=True)
        score_tensor = util.pytorch_cos_sim(english_text_emb, en_ayeh_embedding)
        score_number = score_tensor.item()
        similarity_ayat[key] = score_number
    similarity_ayat = dict(sorted(similarity_ayat.items(), key=lambda item: item[1], reverse=True))
    closest_similarity_ayat = dict()
    for idx, key in enumerate(similarity_ayat):
        similarity = similarity_ayat[key]
        if idx == 10 or similarity < 0.4:
            break
        closest_similarity_ayat[key] = similarity
    return closest_similarity_ayat if len(closest_similarity_ayat) != 0 else None

########################### Ayats and Sourahs ##########################

def __remove_starting_zeros(number_str) -> int:
    real_str = ""
    idx = 0
    for digit in number_str:
        if digit == "0":
            idx += 1
        else:
            break
    return int(number_str[idx:])

def __number_of_ayats_should_pass(surah_number, ayeh_number):
    from constants import surah_ayats_count
    ayeh_count = sum(surah_ayats_count[:surah_number])
    ayeh_count += ayeh_number
    return ayeh_count
    
def find_ayats(position):
    surah = position[:-3]
    ayeh = position[-3:]
    surah_number = __remove_starting_zeros(surah)
    ayeh_number = __remove_starting_zeros(ayeh)
    return surah_number, ayeh_number


def find_ayat_should_be_passed(raw_api_response):
  results = raw_api_response.split("\t")
  ayat_should_be_passed = dict()
  count = 0
  for result in results:
      pos, acc = result.split(':')
      acc = float(acc)
      if acc < 0.7:
          break
      count += 1
      surah_number, ayeh_number = find_ayats(pos)
      ayat_should_be_passed[__number_of_ayats_should_pass(surah_number, ayeh_number)] = count
  total_acceptable_ayat_count = count
  return ayat_should_be_passed, total_acceptable_ayat_count

def find_ayat_should_be_passed_2nd_api(surah_names_with_ayeh_numbers_and_precision: list):
    from constants import surah_ayats_count, surah_name2surah_number
    ayat_should_be_passed = dict()
    count = 1
    for surah_name, ayeh_number, precision in surah_names_with_ayeh_numbers_and_precision:
        if precision < 0.4 or count > 10:
            break
        surah_number = surah_name2surah_number[surah_name]
        ayat_should_be_passed[__number_of_ayats_should_pass(surah_number, ayeh_number)] = count
        count += 1
    total_acceptable_ayat_count = count - 1
    return ayat_should_be_passed, total_acceptable_ayat_count

def find_ayat_text_from_file(ayat_should_be_passed, total_acceptable_ayat_count, farsi_text):
    from constants import EN_AYAT_FILE_NAME, AR_AYAT_FILE_NAME
    ayat_sorted = sorted(ayat_should_be_passed.items(), key=lambda x: x[0])
    ayat_text = [0 for _ in range(total_acceptable_ayat_count)]
    preprocessed_arabic_texts_sorted = copy(ayat_text)
    english_texts_sorted = copy(ayat_text)
    all_preprocessed_arabic_texts = ""
    all_english_texts = ""
    with open(EN_AYAT_FILE_NAME, 'r') as en_file, open(AR_AYAT_FILE_NAME, 'r', encoding='utf8') as ar_file:
        en_line = en_file.readline()
        ar_line = ar_file.readline()
        line_count = 1
        for ayeh_count, _ in ayat_sorted:
            while en_line and ar_line and line_count < ayeh_count:
                en_line = en_file.readline()
                ar_line = ar_file.readline()
                line_count += 1
            surah_no, ayeh_no, en_ayeh_text = en_line.split('|')
            _, _, ar_ayeh_text = ar_line.split('|')
            priorirty_ayeh = ayat_should_be_passed[ayeh_count]
            preprocess_arabic_ayeh = preprocess_ar_ayeh(ar_ayeh_text.strip())
            # all_preprocessed_arabic_texts = all_preprocessed_arabic_texts + "#" + preprocess_arabic_ayeh
            current_english_text = en_ayeh_text.strip() + f" ({surah_no}, {ayeh_no})"
            # all_english_texts = all_english_texts + "#" + current_english_text
            ayat_text[priorirty_ayeh-1] = preprocess_arabic_ayeh + " => " + f"{current_english_text}\n"
            preprocessed_arabic_texts_sorted[priorirty_ayeh-1] = preprocess_arabic_ayeh
            english_texts_sorted[priorirty_ayeh-1] = current_english_text
    all_preprocessed_arabic_texts = "#".join(preprocessed_arabic_texts_sorted)
    all_english_texts = "#".join(english_texts_sorted)
    add_farsi_arabic_text(farsi_text, all_preprocessed_arabic_texts, method=Method.ONE)
    add_farsi_english_text(farsi_text, all_english_texts, method=Method.ONE)
    return ayat_text

def extract_surah_and_ayeh(result):
    """
    Extract surah and ayeh from the result of the data gotten from api
    """
    first_part, second_part = result.split('،')
    surah_name = " ".join(first_part.split()[2:])
    surah_name = unify_arabic_text(surah_name)
    _, second_part, third_part = second_part.split(' ')
    ayeh_number = int(second_part[4:])
    percision = int(third_part[:-1])
    return surah_name, ayeh_number, percision

def init_arabic_and_english_ayat_dict():
    if len(arabic_ayat) != 0 and len(english_ayat) != 0:
        return
    from constants import EN_AYAT_FILE_NAME, AR_AYAT_FILE_NAME
    with open(EN_AYAT_FILE_NAME, 'r') as en_file, open(AR_AYAT_FILE_NAME, 'r', encoding='utf8') as ar_file:
        en_line = en_file.readline()
        ar_line = ar_file.readline()
        while en_line and ar_line:
            surah_no, ayeh_no, en_ayeh_text = en_line.split('|')
            _, _, ar_ayeh_text = ar_line.split('|')
            key = surah_no + "," + ayeh_no
            english_ayat[key] = en_ayeh_text
            arabic_ayat[key] = ar_ayeh_text
            en_line = en_file.readline()
            ar_line = ar_file.readline()   

########################### Preprocessing ###########################

def unify_arabic_text(input_text):
    output_text = input_text.replace('ة', 'ه')
    output_text = output_text.replace('ي', 'ی')
    output_text = output_text.replace('ى', 'ی')     # replace 'ىا' with 'یا'
    output_text = output_text.replace('ؤ', 'و')
    output_text = output_text.replace('أ', 'ا')
    output_text = output_text.replace('إ', 'ا')
    output_text = output_text.replace('ك', 'ک')
    return output_text

def preprocess_en_ayat(ayat_text):
    """
    Remove surah and ayeh number in paranthesis,
    also the first and last ','s.
    """
    import re
    total_ayat_text = []
    ayat_list = ayat_text.split('\n')
    for ayeh_text in ayat_list:
        en_ayeh_text = re.split(" => ", ayeh_text)[-1]
        preproc_ayeh = '('.join(en_ayeh_text.split("(")[:-1]).strip(',')
        total_ayat_text.append(preproc_ayeh)
    return total_ayat_text

def preprocess_ar_ayeh(ayeh_text):
    """
    Remove the Diactrics
    """
    import pyarabic.araby as araby
    preproc_ayeh = araby.strip_diacritics(ayeh_text)
    return preproc_ayeh

def preprocess_arabic_ayat(ayat_text):
    total_ayat_text = []
    ayat_list = ayat_text.split('\n')
    for ayeh_text in ayat_list:
        preproc_ayeh = '('.join(ayeh_text.split("(")[:-1]).strip(',')
        total_ayat_text.append(preproc_ayeh)
    return total_ayat_text

########################### Postprocess ###########################

def postprocess_response(response):
    """
    Remove the English ayat and the "=>" before sending response to frontend.
    """
    import re
    postproc_response = []
    for row in response:
        surah_ayat_no = " (" + row.split('(')[-1]
        arabic_ayeh = re.split(" => ", row)[0]
        postproc_response.append(arabic_ayeh + surah_ayat_no)
    return postproc_response

def remove_not_finished_text(response):
    import re
    last_ending_char_idx = len(response)
    for rev_idx, char in enumerate(response[::-1]):
        if char in '.?!':
            last_ending_char_idx -= rev_idx
            break
    final_response = response[:last_ending_char_idx+1]
    return final_response

def remove_old_text(response, before_gen_len):
    return response[before_gen_len:]

########################### Summarization ###########################

def summarize_english_ayats(preproc_ayats):
    from transformers import pipeline
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    result = summarizer(preproc_ayats, truncation=True)
    return result[0]["summary_text"]

def summarize_arabic_ayat(preproc_ayat):
    from transformers import BertTokenizer, AutoModelForSeq2SeqLM, pipeline
    from arabert.preprocess import ArabertPreprocessor

    model_name="malmarjeh/mbert2mbert-arabic-text-summarization"
    preprocessor = ArabertPreprocessor(model_name="")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipeline = pipeline("text2text-generation",model=model,tokenizer=tokenizer, truncation=True)

    text = preprocessor.preprocess(preproc_ayat)

    result = pipeline(text,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=3,
                repetition_penalty=3.0,
                max_length=200,
                length_penalty=1.0,
                no_repeat_ngram_size = 3)[0]['generated_text']
    return result

########################### English Generation ###########################

def generate_english_text_old(text):
    from transformers import pipeline, set_seed
    text_list = text.split()
    generator = pipeline('text-generation', model='openai-gpt')
    set_seed(42)
    response = generator(text, max_new_tokens=50, min_new_tokens=30)[0]["generated_text"]
    return response

def generate_english_text(text): # change model.generate's max_length to 712
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    encoded_input = tokenizer.encode(text, truncation=True, max_length=512, return_tensors='pt')
    outputs = model.generate(encoded_input, max_length=256, do_sample=True, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

########################### Translation to Arabic ###########################

def translate_en2ar(en_sentence):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    input_ids = tokenizer(en_sentence, return_tensors="pt")
    ar_sentences = model.generate(input_ids=input_ids)
    return tokenizer.batch_decode(ar_sentences, skip_special_tokens=True)
        

def simple_translate_en2ar(en_sentence):
    from transformers import pipeline
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar", truncation=True)
    output = translator(en_sentence)
    return output[0]["translation_text"]

########################### Translation from Farsi to English ###########################

def translate_fa2en(farsi_text):
    from transformers import MT5ForConditionalGeneration, MT5Tokenizer
    model_size = "base"
    model_name = f"persiannlp/mt5-{model_size}-parsinlu-opus-translation_fa_en"
    tokenizer = MT5Tokenizer.from_pretrained(model_name, force_download=True, resume_download=False)
    model = MT5ForConditionalGeneration.from_pretrained(model_name, force_download=True, resume_download=False)
    input_ids = tokenizer.encode(farsi_text, return_tensors="pt")
    res = model.generate(input_ids)
    output = tokenizer.batch_decode(res, skip_special_tokens=True) # Output is in an array (len=1)
    return output[0]

def translate_fa2en_google(farsi_text):
    from googletrans import Translator
    translator = Translator()
    translated_text = translator.translate(farsi_text, src="fa", dest="en").text
    return translated_text
    

########################### Evaluation ###########################

def total_evaluation(input_text, final_arabic_text, method):
    from bert_score import BERTScorer
    from numpy import average
    import torch
    from nltk.translate import bleu as bleu_score
    source_arabic_text = get_arabic_texts(input_text, method=method)[0][0]
    scorer = BERTScorer(model_type="bert-base-multilingual-cased", lang="ar")
    arabic_ayat = source_arabic_text.split('#')
    precisions, recalls, f1_scores, bleus = [], [], [], []
    for src_arabic_ayeh in arabic_ayat:
        if len(src_arabic_ayeh) == 0: continue
        precision, recall, f1_score = scorer.score([final_arabic_text,], [src_arabic_ayeh,])
        bleu = bleu_score([src_arabic_ayeh.split()], final_arabic_text.split())
        precisions.append(precision.item())
        recalls.append(recall.item())
        f1_scores.append(f1_score.item())
        bleus.append(bleu)

    total_src_arabic_ayat = ' '.join(arabic_ayat)
    total_bleu_score = bleu_score([total_src_arabic_ayat.split()], final_arabic_text.split())
    total_precision, total_recall, total_f1_score = [value.item() for value in scorer.score([final_arabic_text,], [total_src_arabic_ayat,])]

    precision_stat = max(precisions), min(precisions), average(precisions), total_precision
    recall_stat = max(recalls), min(recalls), average(recalls), total_recall
    f1_score_stat = max(f1_scores), min(f1_scores), average(f1_scores), total_f1_score
    bleu_score_stat = max(bleus), min(bleus), average(bleus), total_bleu_score
    return precision_stat, recall_stat, f1_score_stat, bleu_score_stat


def bleu_evaluation(input_text, final_arabic_text, method):
    import evaluate
    source_arabic_text = get_arabic_texts(input_text, method)[0][0]
    arabic_ayat: list = source_arabic_text.split('#')
    bleu = evaluate.load("bleu")
    predictions = [final_arabic_text,]
    references = [arabic_ayat,]
    results = bleu.compute(predictions=predictions, references=references)
    return results["bleu"]




if __name__ == "__main__":
    fa_text = "میوه"
    print(translate_fa2en_google(fa_text))
    