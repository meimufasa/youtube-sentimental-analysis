from flask import Flask, render_template, request, url_for
from youtube_comment_downloader import YoutubeCommentDownloader
from sentiments import prediction, ensure_nltk_resources
import os
ensure_nltk_resources()

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
@app.route('/submit', methods=['POST', 'GET'])
def comments_fetcher():
    if request.method == 'POST':
        url = request.form.get('url')
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(url)

        com = {}
        for comment in comments:
            if len(com) >= 10:
                break  # not return
            text = comment['text']
            sentiment = prediction([text])[0]
            com[text] = sentiment

        return render_template('output.html', comments=com)
    else:
        return render_template('index.html')

# comments_fetcher("https://www.youtube.com/watch?v=rdwh9IyF5hc")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)