#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:52:28 2018

@author: sumi
"""

from PIL import Image
from io import BytesIO
from urllib.request  import urlopen
import matplotlib.pyplot as plt



#url = "https://api.nasa.gov/neo/rest/v1/feed?start_date=2015-09-07&end_date=2015-09-08&api_key=DEMO_KEY"
#url = "https://api.nasa.gov/neo/rest/v1/feed?start_date=START_DATE&end_date=END_DATE&api_key=API_KEY"
#url = "https://api.nasa.gov/neo/rest/v1/neo/3542519?api_key=DEMO_KEY"
#url = "https://api.nasa.gov/neo/rest/v1/neo/browse?api_key=DEMO_KEY"
url = "http://maps.googleapis.com/maps/api/staticmap?center=-30.027489,-51.229248&size=800x800&zoom=14&sensor=false"
#url = "https://api.nasa.gov/planetary/apod"

#response = urlopen(url).read()

buffer = BytesIO(urlopen(url).read())

image = Image.open(buffer)

image.show()





















import urllib.request
url = "http://www.google.com/"
request = urllib.request.Request(url)
response = urllib.request.urlopen(request)
print (response.read())



import urllib2
import re
import os
from os.path import basename
from urlparse import urlsplit
from urlparse import urlparse
from posixpath import basename,dirname
 
## function that processes url, if there are any spaces it replaces with '%20' ##
 
def process_url(raw_url):
 if ' ' not in raw_url[-1]:
     raw_url=raw_url.replace(' ','%20')
     return raw_url
 elif ' ' in raw_url[-1]:
     raw_url=raw_url[:-1]
     raw_url=raw_url.replace(' ','%20')
     return raw_url
 
url='' ## give the url here
parse_object=urlparse(url)
dirname=basename(parse_object.path)
if not os.path.exists('images'):
    os.mkdir("images")
os.mkdir("images/"+dirname)
os.chdir("images/"+dirname)
 
urlcontent=urllib2.urlopen(url).read()
imgurls=re.findall('img .*?src="(.*?)"',urlcontent)
for imgurl in imgurls:
 try:
     imgurl=process_url(imgurl)
     imgdata=urllib2.urlopen(imgurl).read()
     filname=basename(urlsplit(imgurl)[2])
     output=open(filname,'wb')
     output.write(imgdata)
     output.close()
     os.remove(filename)
 except:
     pass



