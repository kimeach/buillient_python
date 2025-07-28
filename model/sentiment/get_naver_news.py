import requests
from bs4 import BeautifulSoup

def get_naver_news(query, pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    news_list = []

    for page in range(1, pages + 1):
        url = f"https://search.naver.com/search.naver?where=news&query={query}&start={10 * (page - 1) + 1}"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select("a.link")  # 기사 링크 선택

        print(links)
        for a in links:
            href = a['href']
            if "n.news.naver.com" in href:
                news_list.append(href)

    return list(set(news_list))  # 중복 제거

news_urls = get_naver_news("금융 주식", pages=3)
