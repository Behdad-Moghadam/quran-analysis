BASE_URL = "https://www.alim.org/quran/compare/surah/"
if __name__ == "__main__":
    import requests
    from bs4 import BeautifulSoup
    from constants import TOTAL_SURAH_COUNT, surah_ayats_count, EN_ASAD_FILE_NAME\
                        ,EN_MALIK_FILE_NAME, EN_YUSUF_FILE_NAME, EN_MUSTAFA_FILE_NAME, EN_PIKTAL_FILE_NAME
    for surah_number in range(77, TOTAL_SURAH_COUNT+1):
        for ayeh_number in range(1, surah_ayats_count[surah_number]+1):
            current_url = BASE_URL + f"{surah_number}/{ayeh_number}/"
            raw_page = requests.get(current_url)
            response_content = raw_page.text
            soup = BeautifulSoup(response_content, 'html.parser')
            translation_elements = soup.findAll('div', class_="row trans-row")[:-1]
            for translation_element in translation_elements:
                translator = translation_element.find('div', class_='col-sm-5').text.split()[0]
                contents = translation_element.find('div', class_='col-sm-7').contents
                raw_translation = ''
                for content in contents:
                    if content.name is None:            # It's a NavigableString
                        raw_translation += content.text
                    elif content.name == "span":
                        continue
                    elif content.name == "div":
                        break
                translation = ' '.join(raw_translation.split())
                content = f"{surah_number}|{ayeh_number}|{translation}\n"
                filename = ''
                if translator == "Asad":
                    filename = EN_ASAD_FILE_NAME
                elif translator == "Malik":
                    filename = EN_MALIK_FILE_NAME
                elif translator == "Yusuf":
                    filename = EN_YUSUF_FILE_NAME
                elif translator == "Mustafa":
                    filename = EN_MUSTAFA_FILE_NAME
                elif translator == "Piktal":
                    filename = EN_PIKTAL_FILE_NAME
                with open(filename, 'a', encoding='utf-8') as f:
                    f.write(content)
            print(f"{surah_number}|{ayeh_number} DONE!")