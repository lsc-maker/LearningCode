import requests
url = "https://m.ip138.com/iplookup.asp?ip="
try:
	r = requests.get(url+'202.96.209.133')
	r.raise_for_status()
	r.encooding = r.apparent_encoding
	print(r.text[-500:])
except:
	print('爬取失败')