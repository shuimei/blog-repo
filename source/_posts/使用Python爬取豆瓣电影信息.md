---
title: 使用Python爬取豆瓣电影信息
---
最近打算做做数据抓取练练手，也收集一些网络数据，于是从豆瓣电影开始尝试抓取电影信息。经过几天的运行，现在已经获取了从豆瓣上记录的从1880年到2015年的约39380部电影的主要信息。这个小项目中没有用到特别应对反爬虫机制，反正机器有的是时间，cookies也是手动替换的。获取到数据是第一步， 接下来还要对数据进行一些可视化分析，做一个酷炫的展示网页就喜闻乐见啦。
项目地址：[shuimei/douban-movie-crawler](shuimei/douban-movie-crawler)

## 定义url

豆瓣电影的站点url规则非常简单，每部电影对应唯一的subject值，这也是url中的一个参数。我是按照年份来收集电影的url的，按照年份，如2015年的url就是：


``` bash
https://movie.douban.com/tag/2015
```
上面的url可定位到2015年电影记录的首页，如果请求翻页，可以添加一个“start”参数，表示这个页面第一部电影的编号，如：

``` bash
https://movie.douban.com/tag/2015?start=100&type=T

```
表示从第100部电影开始记录。

使用该规则，可以完成翻页动作

## 获取所有url
![https://pic2.zhimg.com/v2-847cc0006dbac17457ddc30dd37271d5_b.png](https://pic2.zhimg.com/v2-847cc0006dbac17457ddc30dd37271d5_b.png)

这是按年份索引的电影列表页面，在这个页面我们只需要获取电影名称和电影页面的url即可，使用requests和lxml模块可以完成这个简单的数据提取任务：

``` Python
from lxml import etree
import requests
headers= { 'User-Agent' : 'User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36' }
start_url = "https://movie.douban.com/tag/2015?start=100&type=T"
html = requests.get(start_url,headers=headers).content
selector = etree.HTML(html)
#xpath
names = selector.xpath("//div[@class='pl2']/a/text()")
links = selector.xpath("//div[@class='pl2']/a/@href")
```
再把name和links都写入到文件中。

得到所有电影的url后，就可以继续访问每部电影的主页面，从而获取更多信息。在这个项目中，主要关注电影的以下信息：

+ subject: 电影唯一标识
+ name: 电影名称
+ year: 发行年份
+ directors: 导演
+ actors: 主演
+ release_date: 上映日期
+ star: 豆瓣评分
+ rating_peoplr: 评分人数
+ genres: 电影类型
+ awards: 电影主要获得的奖项
+ image_src: 电影海报链接
+ tags: 主要标签

!()[https://pic2.zhimg.com/v2-6ca37a41a987ffbb4c2924c12fdc4ec9_b.png]
这些信息主要集中在页面的这个板块中。
使用xpath可以方便地对这些信息进行提取：

``` Python
def getMovieInfo(name, url):
	headers= { 'User-Agent' : 'User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36' }
	html = requests.get(url,headers=headers).content.decode("utf-8")
	selector = etree.HTML(html)
	strCat = lambda x,y:x+"/"+y
	# movie subject
	subject = url.split("/")[-2]
	# movie name
	tmp = selector.xpath("//h1/span[@property='v:itemreviewed']/text()")
	name = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# movie release year
	tmp = selector.xpath("//h1/span[@class='year']/text()")
	year = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# movie director(s)
	tmp = selector.xpath("//a[@rel='v:directedBy']/text()")
	directors = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# movie actor(s)
	tmp = selector.xpath("//a[@rel='v:starring']/text()")
	actors = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# movie release date
	tmp = selector.xpath("//span[@property='v:initialReleaseDate']/text()")
	date = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# movie runtime
	tmp = selector.xpath("//span[@property='v:runtime']/text()")
	time = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# movie rating by douban site
	tmp = selector.xpath("//strong[@class='ll rating_num']/text()")
	star = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# number of rating people
	tmp = selector.xpath("//span[@property='v:votes']/text()")
	rating_people = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# movie genre
	tmp = selector.xpath("//span[@property='v:genre']/text()")
	genres = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# movie award
	tmp = selector.xpath("//ul[@class='award']/li/text() | //ul[@class='award']/li/a/text()")
	awards = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp).replace("\n","")
	awards = awards.replace(" ","")
	# str_awards = reduce(strCat, awards)
	# movie post image url
	tmp = selector.xpath("//a[@class='nbgnbg']/img[@rel='v:image']/@src")
	image_src = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# # movie introduction
	# tmp = selector.xpath("//span[@property='v:summary']/text()")
	# introduction =  len(tmp) == 0 and "NotDefined" or tmp[0]
	# str_introduction = reduce(strCat, introduction)
	# movie common tags
	tmp = selector.xpath("//div[@class='tags-body']/a/text()")
	tags = len(tmp) == 0 and "NotDefined" or reduce(strCat, tmp)
	# summary
	movie_info = {
		"subject":subject,
		"name": name,
		"year": year,
		"directors": directors,
		"actors": actors,
		"release_date": date,
		"runtime": time,
		"star": star,
		"rating_people": rating_people,
		"genres": genres,
		"awards": awards,
		"image_src": image_src,
		# "introduction": str_introduction,
		"tags": tags,
	}
	return movie_info
```
最后将这些信息输出到文件。