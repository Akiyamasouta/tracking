from pytube import YouTube

url = input("ダウンロードしたい動画のURL:")
print(*YouTube(url).streams.all(), sep="\n")
itag = int(input("ダウンロードしたい動画のタグ:"))
YouTube(url).streams.get_by_itag(itag).download(r"./video/")