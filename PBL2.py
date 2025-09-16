import feedparser
import pandas as pd
from gtts import gTTS

# 기사 제목/요약 순서로 txt 파일 만들기
def make_txt(url):
    feed = feedparser.parse(url)

    titles = []
    descriptions = []

    for entry in feed.entries:
        titles.append(entry.title)
        descriptions.append(entry.description)

    with open("resultText.txt", 'w', encoding='utf-8') as file:
        for title, description in zip(titles, descriptions):
            file.write(f"기사제목 {title}\n")
            file.write(f"요약내용 {description}\n")

# mp3 파일 만들기
def make_mp3():
    # 파일 읽기
    with open("resultText.txt", "r", encoding="utf-8") as file:
        text = file.read()

    tts = gTTS(text=text, lang='ko', slow=False)
    tts.save('result.mp3')

# 메인
if __name__ == '__main__':
    url = "https://www.dailysecu.com/rss/allArticle.xml" # 요약 대상
    make_txt(url)
    make_mp3()