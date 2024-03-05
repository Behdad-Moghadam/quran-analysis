BASE_URL = "https://surahquran.com/Surah-translation/meanings-fa-language-20-surah-"
if __name__ == "__main__":
    import requests
    from bs4 import BeautifulSoup
    from constants import TOTAL_SURAH_COUNT, FA_QURAN_FILE_NAME
    for surah_number in range(9, TOTAL_SURAH_COUNT+1):
        current_url = BASE_URL + f"{surah_number}.html"
        raw_page = requests.get(current_url)
        response_content = raw_page.text
        soup = BeautifulSoup(response_content, 'html.parser')
        translation_elements = soup.findAll('a', class_="her")
        with open(FA_QURAN_FILE_NAME, 'a', encoding='utf-8') as fa_quran_file:
            for idx, translation_element in enumerate(translation_elements):
                ayeh_number = idx + 1
                translation = translation_element.text
                content = f"{surah_number}|{ayeh_number}|{translation}\n"
                fa_quran_file.write(content)
                print(f"{surah_number}|{ayeh_number} DONE!")