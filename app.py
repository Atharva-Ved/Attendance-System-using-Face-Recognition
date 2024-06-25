import cv2
import os
from flask import Flask, render_template, request, redirect, session, send_file, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import mysql.connector
from flask_cors import CORS
import pickle



# Defining Flask App
app = Flask(__name__)
app.secret_key = 'Mykey'
CORS(app)

nimgs = 1
known_face_encodings=[]
# Connect to MySQL database
# connection = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='',
#     database='Recognition'
# )

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%d_%m_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Date')
if not os.path.isdir('static/voices'):
    os.makedirs('static/voices')



# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
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


# A function which trains the model on all the faces available in faces folder
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


# Extract info from today's attendance file in attendance folder
import pandas as pd

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    
   # Calculate attendance percentage for each user
    attendance_counts = df['Name'].value_counts()
    total_sessions = l  # Total number of sessions
    attendance_percentage = {}
    for name, count in attendance_counts.items():
        attendance_percentage[name] = (count /1) * 100

    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name,time,branch,div):
    time= time.replace(":", "-")
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    print(time)
    if f'Attendance/{branch}-{div}-{datetoday}-{time}.csv' not in os.listdir('Attendance'):
        print("creating csv")
        with open(f'Attendance/{branch}-{div}-{datetoday}-{time}.csv', 'w') as f:
            f.write('Name,Roll,Time,Date')
        print("created csv")

    df = pd.read_csv(f'Attendance/{branch}-{div}-{datetoday}-{time}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/{branch}-{div}-{datetoday}-{time}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l= len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)

#generate encodings for image and store them
def generate_encoding() :
    folderPath = 'static/Faces'
    imgPathList = os.listdir(folderPath)
    imgList = []
    encodeList = []
    studentIds = []
    for path in imgPathList:
        imgList.append(cv2.imread(os.path.join(folderPath,path)))
        studentIds.append(os.path.splitext(path)[0])

    # generate incoding
    for img in imgList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    encodingListWithIds = [encodeList, studentIds]    
    print(encodingListWithIds)
    # save encoding to pickle file
    filename = "Encodings.p"
    if not os.path.exists(filename):
        open(filename,'wb').close()

    file = open(filename,'ab')
    pickle.dump(encodingListWithIds, file)
    file.close()

#load the encoding file
KnownStudentIds = []
knownEncodeList = []

def load_encoding():
    filename = "Encodings.p"
    file = open(filename,'rb')
    encodeListwithIds = pickle.load(file)
    knowEncodeList, KnownStudentIds = encodeListwithIds

################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/', methods=['GET', 'POST'])
def login_redirect():
    return render_template('dashboard.html')

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
                return render_template('dashboard.html', username=session['username'], names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
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
    # if 'logged_in' in session:
    #     # names, rolls, times, l = extract_attendance()
    #     username = session['username']
        return render_template('take_attendance.html')#, username=username)
    # else:
        # return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    # names, rolls, times, l = extract_attendance()
    userlist, names, rolls, l = getallusers()
    return render_template('dashboard.html', names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)



## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('userlist.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# @app.route('/roll-call')
# def roll_call():
#     # Assuming you have some data to pass to the template, modify this accordingly
#     return render_template('roll-call.html')


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET',  'POST'])
def start():
    div=request.form['div']
    time=request.form['time_slot']
    branch=request.form['branch']
    
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, img = cap.read()
        cap.set(3, 440)
        cap.set(4, 380)
        imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgSmall)
        if faceCurFrame:  # Check if any faces are detected in the current frame
            encodeCurFrame = face_recognition.face_encodings(imgSmall, faceCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(knownEncodeList, encodeFace)
                faceDis = face_recognition.face_distance(knownEncodeList, encodeFace)
                print(matches)
                if faceDis:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        name = KnownStudentIds[matchIndex]
                        name = name.split('_', 1)[1]

                        add_attendance(name, time, branch, div)
        cv2.imshow('Attendance', img)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return render_template('take_attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/addUser')
def addUser():
    return render_template('addUser.html')
# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    user_image = request.files['file']
    if user_image.filename == '':
        return render_template('addUser.html', mess='No selected file')
    
    userimagefolder = os.path.join('static', 'faces')
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    # Generate a secure filename for the uploaded file
    img_name = f"{newuserid}_{newusername}.jpg"  # or any other image format you want to use
    # Save the uploaded file to the user's folder
    user_image.save(os.path.join(userimagefolder, img_name))
    
    print('encoding started')
    generate_encoding()
    print('encoding complete')
    names, rolls, times, l = extract_attendance()
    return render_template('addUser.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/attendanceList')
def attndanceList():
    return render_template('attendanceList.html')

@app.route('/csv', methods=['GET', 'POST'])
def download_csv():
    branch=request.form['branch']
    div=request.form['div']
    csv_files = [f for f in os.listdir('Attendance') if f.startswith(f'{branch}-{div}')]
    csv_files = sorted(csv_files)
    return render_template('csv.html', csv_files=csv_files)

@app.route('/download/<csv_file>')
def download(csv_file):
    csv_file=f'Attendance/{csv_file}'
    return send_file(csv_file, as_attachment=True)

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True,port=5100)