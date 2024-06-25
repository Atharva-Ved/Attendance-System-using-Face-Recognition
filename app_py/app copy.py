from flask import Flask, render_template, request, redirect, session
from datetime import date, datetime, timedelta
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import mysql.connector
from flask_cors import CORS

# Defining Flask App
app = Flask(__name__)
app.secret_key = 'Mykey'
CORS(app)

nimgs = 10

# Initialize face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize attendance directory
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('static/voices'):
    os.makedirs('static/voices')

# Connect to MySQL database
def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='Recognition'
        )
        return connection
    except mysql.connector.Error as error:
        print("Error:", error)
        return None

# Save date today in different formats
datetoday = date.today()
datetoday_str = datetoday.strftime("%m_%d_%y")
datetoday2 = datetoday.strftime("%d-%B-%Y")

# Generate attendance file name based on current time slot
def generate_attendance_filename():
    current_time_slot = datetime.now().strftime("%H:%M")
    attendance_filename = f'Attendance_{datetoday_str}_{current_time_slot}.csv'
    return attendance_filename

# Check if attendance file exists for current time slot, create if not
def check_create_attendance_file():
    attendance_filename = generate_attendance_filename()
    if attendance_filename not in os.listdir('Attendance'):
        with open(f'Attendance/{attendance_filename}', 'w') as f:
            f.write('Name,Roll,Time,Date')
    return attendance_filename

# Extract faces from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Train the face recognition model
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract info from today's attendance file
def extract_attendance():
    attendance_filename = generate_attendance_filename()
    if attendance_filename in os.listdir('Attendance'):
        df = pd.read_csv(f'Attendance/{attendance_filename}')
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)

        # Calculate attendance percentage for each user
        attendance_counts = df['Name'].value_counts()
        total_sessions = l  # Total number of sessions
        attendance_percentage = {}
        for name, count in attendance_counts.items():
            attendance_percentage[name] = (count / total_sessions) * 100

        return names, rolls, times, l, attendance_percentage
    else:
        return [], [], [], 0, {}

# Add attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    attendance_filename = generate_attendance_filename()
    df = pd.read_csv(f'Attendance/{attendance_filename}')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/{attendance_filename}', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{datetoday}')

# Get names and roll numbers of all users
def get_all_users():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

# Delete a user folder
def delete_folder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)

# Routing Functions
@app.route('/', methods=['GET', 'POST'])
def login_redirect():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='',
                database='Recognition'
            )
            cursor = connection.cursor()
            cursor.execute("SELECT id, Username FROM Login WHERE username = %s AND password = %s", (username, password))
            user = cursor.fetchone()
            print(user)
            cursor.close()
            # if username in users and users[username] == password:
            if user:
                session['logged_in'] = True
                session['username'] = user[1]
                names, rolls, times, l = extract_attendance()
                userlist, names, rolls, l = getallusers()
                return render_template('dashboard.html', username=session['username'], names=names, rolls=rolls, times=times, l=l, datetoday2=datetoday2)
            else:
                return render_template('login.html', error='Invalid username or password')
            
        except mysql.connector.Error as error:
            print("Error:", error)
            return render_template('login.html', error='Error connecting to database')

        finally:
            # Close the cursor and connection
            if 'connection' in locals():
                cursor.close()
                connection.close()

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return render_template('login.html')

@app.route('/TakeAttendance')
def home():
    attendance_filename = check_create_attendance_file()
    names, rolls, times, l, attendance_percentage = extract_attendance()
    return render_template('take_attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/dashboard')
def dashboard():
    names, rolls, times, l, attendance_percentage = extract_attendance()
    userlist, names, rolls, l = get_all_users()
    return render_template('dashboard.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = get_all_users()
    return render_template('userlist.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    delete_folder('static/faces/'+duser)
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    try:
        train_model()
    except:
        pass
    userlist, names, rolls, l = get_all_users()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    attendance_filename = check_create_attendance_file()
    names, rolls, times, l, attendance_percentage = extract_attendance()
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l, attendance_percentage = extract_attendance()
    return render_template('take_attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/addUser')
def addUser():
    return render_template('addUser.html')

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l, attendance_percentage = extract_attendance()
    return render_template('addUser.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Main Function
if __name__ == '__main__':
    app.run(debug=True, port=5100)
