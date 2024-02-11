from flask import Flask, render_template, request
from in_out import in_out
from motion import noise
from rect_noise import rect_noise
from record import record
from PIL import Image, ImageTk
from find_motion import find_motion
from spot_diff import spot_diff

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/in_out', methods=['GET', 'POST'])
def in_out_web():
    if request.method == 'POST':
        # Do the in_out function
        in_out()
    return render_template('in_out.html')

@app.route('/noise', methods=['GET', 'POST'])
def noise_web():
    if request.method == 'POST':
        # Do the noise function
        noise()
    return render_template('noise.html')

@app.route('/rect_noise', methods=['GET', 'POST'])
def rect_noise_web():
    if request.method == 'POST':
        # Do the rect_noise function
        rect_noise(use_time_threshold=True) 
    return render_template('rect_noise.html')

@app.route('/record', methods=['GET', 'POST'])
def record_web():
    if request.method == 'POST':
        # Do the record function
        record()
    return render_template('record.html')

@app.route('/find_motion', methods=['GET', 'POST'])
def find_motion_web():
    if request.method == 'POST':
        # Do the find_motion function
        find_motion()
    return render_template('find_motion.html')

if __name__ == '__main__':
    app.run(debug=True)
