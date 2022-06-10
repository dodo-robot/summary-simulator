#!/usr/bin/env python
""" 
Author: Giorgio Raineri
"""

# Import standard modules
from urllib.parse import urlparse
import re

# Import Libraries
import newspaper.article
from newspaper import Article
from requests import get

# Import Application Modules
from simulator.logger.Logger import logger


class SmartScraperArticle:
    """
    A wrapper for the Article class of the Newspaper library.
    """
    def __init__(self, article: Article):
        self.title = article.title
        self.subtitle = ""
        self.body = article.text
        self.url = article.url
        self.source = article.source_url
        self.date = article.publish_date

        self.authors = []
        for author in article.authors:
            clean_name = re.sub(r'^di ', '', author, flags=re.IGNORECASE)
            self.authors.append(clean_name)

        self.tags = set()

        for keyword in article.keywords:
            self.tags.add(keyword)

        for tag in article.tags:
            self.tags.add(tag)

        for keyword in article.meta_keywords:
            self.tags.add(keyword)

        self.cover_image = article.meta_img
        self.images_list = article.images

    def get_NiFi_version(self):
        if self.body != '':
            return {
                "news_title": self.title,
                "news_subtitle": self.subtitle,
                "body": self.body,
                "url": self.url,
                "source": urlparse(self.source).netloc,
                "tags": list(self.tags),
                "cover_image": self.cover_image,
                "images_list": list(self.images_list),
                "authors": self.authors,
                "is_generic_scraper": True
            }
        else:
            return None


class SmartScraper:
    """
    A scraper to extract the content of a generic web page.
    """
    def __init__(self, url):
        """
        Initializes the old_scrapers.
        :param url: The url of the article to be scraped.
        """
        r = get(url)
        self.url = r.url
        self.parser = Article(self.url)
        self.parsed_article = None

    def __parse_article(self):
        """
        Downloads the article, parses it and stores it in a SmartScraperArticle object.
        """
        try:
            self.parser.download()
            self.parser.parse()
            self.parsed_article = SmartScraperArticle(self.parser)
        except newspaper.article.ArticleException as e:
            logger.error(f"Error while parsing article: {e}")
            self.parsed_article = None

    def get_article(self):
        """
        Returns an object with the content of the article.
        :return: A SmartScraperArticle object.
        """
        self.__parse_article()
        return self.parsed_article.get_NiFi_version() if self.parsed_article else None


if __name__ == '__main__':
    url = "https://www.ilfattoquotidiano.it/2022/04/07/guerra-russia-ucraina-lanalisi-del-generale-bertolini-putin-si-riposiziona-verso-i-reali-obiettivi-kiev-diversivo-tattico-e-politico/6551253/"
    scraper = SmartScraper(url)
    print(scraper.get_article())