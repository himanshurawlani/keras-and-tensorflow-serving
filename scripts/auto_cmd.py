import os
import signal
import subprocess

# Making sure to use virtual environment libraries
activate_this = "/home/ubuntu/tensorflow/bin/activate_this.py"
exec(open(activate_this).read(), dict(__file__=activate_this))

# Change directory to where your Flask's app.py is present
os.chdir("/home/ubuntu/Desktop/Medium/keras-and-tensorflow-serving/flask_server")
tf_ic_server = ""
flask_server = ""

try:
    tf_ic_server = subprocess.Popen(["tensorflow_model_server "
                                     "--model_base_path=/home/ubuntu/Desktop/Medium/keras-and-tensorflow-serving/my_image_classifier "
                                     "--rest_api_port=9000 --model_name=ImageClassifier"],
                                    stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("Started TensorFlow DamageAnalyzer server!")

    flask_server = subprocess.Popen(["export FLASK_ENV=development && flask run --host=0.0.0.0"],
                                    stdout=subprocess.DEVNULL,
                                    shell=True,
                                    preexec_fn=os.setsid)
    print("Started Flask server!")

    while True:
        print("Type 'exit' and press 'enter' to quit: ")
        in_str = input().strip().lower()
        if in_str == 'q' or in_str == 'exit':
            print('Shutting down all servers...')
            os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
            os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
            print('Servers successfully shutdown!')
            break
        else:
            continue
except KeyboardInterrupt:
    print('Shutting down all servers...')
    os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
    os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
    print('Servers successfully shutdown!')
