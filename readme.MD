# Real-time sign language detection system in video calling web-app

![Alt text](https://github.com/devarsh-mavani-19/RealtimeSignLanguageRecognition/blob/master/image.png?raw=true)

# Tech stack
`Backend:`
1. `NodeJS`
2. `ExpressJS`
3. `Socket.io (Websockets)`
4. `PeerJS`

`Frontend`
1. `EJS`
2. `HTML/CSS/JS`

`Machine Learning`
1. `Mediapipe`
2. `TensorflowJS`

# Installation
1. fork the repo
2. `git clone <repo-name>`
3. `cd RealtimeSignLanguageRecognition`
4. `npm install`
5. `npm start`


# How to add new data
1. Go to ML Folder and open `collect-dataset.py`
2. edit line#97 `mdict['sign'] = 'G'` and replace the symbol G
3. Run the script using `python collect-dataset.py`
4. press `t` to capture image and `q` to stop
5. Run `python predict.py` file to export model in `json` format
6. copy model and bin file to static/model folder
