from flask import Flask, render_template, request, send_file
from gtts import gTTS
from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip
import os
import uuid

app = Flask(__name__)
VIDEO_FOLDER = 'static/videos/'

if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        filename = str(uuid.uuid4())

        # Create audio
        tts = gTTS(text=text, lang='en')
        audio_path = os.path.join(VIDEO_FOLDER, f'{filename}.mp3')
        tts.save(audio_path)

        # Create video
        clip = TextClip(text, fontsize=50, color='white', size=(720, 480))
        clip = clip.set_duration(5)
        audio_clip = AudioFileClip(audio_path)
        video = clip.set_audio(audio_clip)
        video_path = os.path.join(VIDEO_FOLDER, f'{filename}.mp4')
        video.write_videofile(video_path, fps=24)

        return render_template('index.html', video_file=f'videos/{filename}.mp4')

    return render_template('index.html', video_file=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))

