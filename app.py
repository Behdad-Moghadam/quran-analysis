from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def closest_ayats1():
    from main import find_closest_ayat_2, find_closest_ayat_3
    if request.method == 'POST':
        text = request.form['text']
        selected_radio_btn = request.form['radioButton']
        find_closest_ayat_func = None
        if selected_radio_btn == "method1":
            find_closest_ayat_func = find_closest_ayat_2
        elif selected_radio_btn == "method2":
            find_closest_ayat_func = find_closest_ayat_3
        closest_ayat = find_closest_ayat_func(text)
        return jsonify({'response': closest_ayat})
    return render_template('main.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_ayats():
    from main import preprocess_en_ayat
    text = request.form['text']
    response = ' '.join(preprocess_en_ayat(text))
    return jsonify({'response': response})

@app.route('/summarize', methods=['POST'])
def summarize_ayats():
    from main import summarize_english_ayats
    preproc_text = request.form['text']
    response = summarize_english_ayats(preproc_text)
    return jsonify({'response': response})

@app.route('/translate', methods=['POST'])
def translate_to_arabic():
    from main import simple_translate_en2ar
    en_text = request.form['text']
    response = simple_translate_en2ar(en_text)
    return jsonify({'response': response})

@app.route('/evaluate', methods=["POST"])
def evaluate():
    from main import total_evaluation
    print("evaluate test")
    input_text = request.form["input"]
    print("input_text:", input_text)
    final_arabic_text = request.form["text"]
    print("final_arabic_text:", final_arabic_text)
    method_name = request.form['radioButton']
    print("method_name:", method_name)
    precision_stat, recall_stat, f1_score_stat, bleu_score_stat = total_evaluation(input_text, final_arabic_text, method_name)
    return jsonify({
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
    })

########################### Arabic Summarization ###########################

@app.route('/arsum', methods=["GET", "POST"])
def closest_ar_ayat():
    from main import find_closest_ayat_2, find_closest_ayat_3, postprocess_response
    if request.method == 'POST':
        text = request.form['text']
        selected_radio_btn = request.form['radioButton']
        find_closest_ayat_func = None
        if selected_radio_btn == "method1":
            find_closest_ayat_func = find_closest_ayat_2
        elif selected_radio_btn == "method2":
            find_closest_ayat_func = find_closest_ayat_3
        closest_ayat = find_closest_ayat_func(text)
        arabic_closest_ayat = postprocess_response(closest_ayat)
        return jsonify({'response': arabic_closest_ayat})
    return render_template('arabic_summary.html')

@app.route('/arsum/summarize', methods=["POST"])
def arabic_summarization():
    from main import preprocess_arabic_ayat, summarize_arabic_ayat
    text = request.form['text']
    ayat = preprocess_arabic_ayat(text)
    summary = summarize_arabic_ayat(ayat)
    return jsonify({'response': summary})

########################### Generation ###########################

@app.route('/generation', methods=["GET", "POST"])
def ar_en_closest_ayat():
    from main import find_closest_ayat_2, find_closest_ayat_3
    if request.method == 'POST':
        text = request.form['text']
        selected_radio_btn = request.form['radioButton']
        find_closest_ayat_func = None
        if selected_radio_btn == "method1":
            find_closest_ayat_func = find_closest_ayat_2
        elif selected_radio_btn == "method2":
            find_closest_ayat_func = find_closest_ayat_3
        closest_ayat = find_closest_ayat_func(text)
        return jsonify({'response': closest_ayat})
    return render_template('arabic_generation.html')

@app.route('/generation/preprocess', methods=["POST"])
def preprocess_generation():
    return preprocess_ayats()

@app.route('/generation/generate', methods=["POST"])
def english_generation():
    from main import generate_english_text, remove_old_text, remove_not_finished_text
    if request.method == 'POST':
        text = request.form['text']
        generated_text = generate_english_text(text)
        postproc_text = remove_old_text(generated_text, len(text))
        postproc_text = remove_not_finished_text(postproc_text)
        return jsonify({'response': postproc_text})

@app.route('/generation/translate', methods=["POST"])
def translate_generation():
    return translate_to_arabic()

if __name__ == '__main__':
    app.run()
