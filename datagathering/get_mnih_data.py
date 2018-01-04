'''
Crawl all images from Mnih's webpage
using beautiful soup as html-parser for the links
'''

from bs4 import BeautifulSoup
import os
import urllib.request



# set folder for storage
storage = "/home/jgucci/Desktop/mnih_data/"

# get all 3 data sets (train, valid, test)
datasets = ['train', 'valid', 'test']

for data in datasets:

    for type in ["sat", "map"]:

        # create storage folder
        storagefolder = storage + data + "/" + "raw/"

        if not os.path.exists(storagefolder):
            os.makedirs(storagefolder)

        # get specific url
        url = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/" + data + "/" + type + "/index.html"

        # get html code
        with urllib.request.urlopen(url) as sourcecode:
            mystr = sourcecode.read().decode("utf8")

        # parse
        soup = BeautifulSoup(mystr, 'html.parser')
        # soup.find_all('a')
        # create list of all downloadable links
        linklist = []

        # retrieve al links on the homepage
        for link in soup.find_all('a'):
        #print(link.get('href'))
            linklist.append((link.get('href')))

        # counter for image storage
        counter = 0

        # inform user
        print("downloading " + data + " " + " " + type)

        # retrieve images from linklist
        for link in linklist:
            if counter <= 400:
                urllib.request.urlretrieve(url = link, filename = storagefolder + type + str(counter) +'.tiff')
                counter += 1


