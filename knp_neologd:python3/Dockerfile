FROM python:3-stretch
MAINTAINER Runa Yamada

WORKDIR /root

ENV DEBIAN_FRONTEND noninteractive

# JUMAN KNP Version
ENV JUMANPP_VERSION 1.02
ENV JUMAN_VERSION 7.01
ENV KNP_VERSION 4.19

# proxy settings
ENV https_proxy "https://proxy.kwansei.ac.jp:8080"
ENV http_proxy "http://proxy.kwansei.ac.jp:8080"

RUN apt-get update --fix-missing &&\
    apt-get install -y --fix-missing apt-utils dialog
RUN apt-get upgrade -y --fix-missing &&\
    apt-get install -y --fix-missing libboost-dev &&\
    apt-get install -y --fix-missing google-perftools &&\
    apt-get install -y --fix-missing libgoogle-perftools-dev &&\
# python container had already
    apt-get install -y --fix-missing gcc &&\
    apt-get install -y --fix-missing g++ &&\
    apt-get install -y --fix-missing make &&\
    apt-get install -y --fix-missing wget &&\
# to decompress *.tar.bz2
    apt-get install -y --fix-missing bzip2 &&\

# install JUMAN
    wget http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/juman/juman-${JUMAN_VERSION}.tar.bz2 -O /tmp/juman.tar.bz2 &&\ 
    tar xf /tmp/juman.tar.bz2 -C /tmp &&\
    cd /tmp/juman-${JUMAN_VERSION} &&\ 
    ./configure --prefix=/usr/local/ && make && make install &&\
    # rm -rf /tmp/* &&\
    rm -rf /var/cache/apk/* &&\
    apt-get update && apt-get install -y --fix-missing libjuman4 &&\
# install KNP
    apt-get install -y --fix-missing zlib1g-dev &&\
    wget http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/knp/knp-${KNP_VERSION}.tar.bz2 -O /tmp/knp.tar.bz2 &&\
    tar xf /tmp/knp.tar.bz2 -C /tmp &&\
    cd /tmp/knp-${KNP_VERSION} / &&\
    ./configure --prefix=/usr/local/ --with-juman-prefix=/usr/local/ && make && make install &&\
    # rm -rf /tmp/* &&\
    rm -rf /var/cache/apk/*

# pip install
	RUN pip install --upgrade pip
	RUN pip install --upgrade setuptools
	RUN pip install six
	RUN pip install pyknp
	RUN pip install tqdm
	RUN pip install gensim
	RUN pip install lxml
	RUN pip install matplotlib
	RUN pip install nltk
	RUN pip install xlrd
	RUN pip install sklearn
	RUN pip	install jaconv

# to use Japanese
RUN apt-get install -y -f locales
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8
RUN localedef -f UTF-8 -i ja_JP ja_JP.utf8

# clean up all temporary files 
RUN apt-get clean &&\
    apt-get autoclean -y &&\
    apt-get autoremove -y &&\
    apt-get clean &&\
    # rm -rf /tmp/* /var/tmp/* &&\
    rm -rf /var/lib/apt/lists/* &&\    
    rm -f /etc/ssh/ssh_host_*
	
CMD /bin/bash

# to use in matplotlib

RUN apt install unzip
RUN wget -O ipafont.zip http://ipafont.ipa.go.jp/old/ipafont/IPAfont00303.php
RUN unzip ipafont.zip
RUN mv IPAfont00303/*.ttf /usr/local/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/
# RUN .cache/matplotlib/fontList.cache
