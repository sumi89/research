#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:56:57 2018

@author: sumi
"""

import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
from io import BytesIO
from urllib.request  import urlopen


#url = "https://apod.nasa.gov/apod/image/1504/Mooooonwalk_rjn_960.jpg"

api = "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY&date=2015-04-01"


response = requests.get(api)
# parse html
page = str(BeautifulSoup(response.content))

url = re.search("(?P<url>https?://[^\s]+)", page).group("url")

url = re.findall(r'(https?://[^\s]+)', page)


buffer = BytesIO(urlopen(url).read())

image = Image.open(buffer)

image.show()










def getURL(page):
    """

    :param page: html of web page (here: Python home page) 
    :return: urls in that page 
    """
    start_link = page.find("a href")
    if start_link == -1:
        return None, 0
    start_quote = page.find('"', start_link)
    end_quote = page.find('"', start_quote + 1)
    url = page[start_quote + 1: end_quote]
    return url

