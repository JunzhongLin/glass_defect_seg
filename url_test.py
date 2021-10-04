from urllib.request import urlopen

url = 'http://hq.sinajs.cn/list=sh600000'
request = urlopen(url)
content = request.read()
content = content.decode('gbk')
print(content)