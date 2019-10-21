import cv2
from time import time
from mtcnn_detector import MtcnnDetector
import kcftracker
import mxnet as mx
import numpy as np
from sklearn.decomposition import FastICA, PCA
import scipy as scipy
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

selectingObject = False
initTracking = True
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
cutlow, w, h = 0, 0, 0
prevout, out6, previous = 0.0, 60.81, 70
inteval = 1
duration = 0.01
meanlist = []
meanlistKF = []
result = []
trackerm = []
ROIS = []

#creates a butter pass filter
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#creates a butter pass filter by cutting off certain frequency
#returns filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def running_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0))
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
	global selectingObject, initTracking, onTracking, ix, iy, cx,cy, w, h
	
	if event == cv2.EVENT_LBUTTONDOWN:
		selectingObject = True
		onTracking = False
		ix, iy = x, y
		cx, cy = x, y
	
	elif event == cv2.EVENT_MOUSEMOVE:
		cx, cy = x, y
	
	elif event == cv2.EVENT_LBUTTONUP:
		selectingObject = False
		if(abs(x-ix)>10 and abs(y-iy)>10):
			w, h = abs(x - ix), abs(y - iy)
			ix, iy = min(x, ix), min(y, iy)
			initTracking = True
		else:
			onTracking = False
	
	elif event == cv2.EVENT_RBUTTONDOWN:
		onTracking = False
		if(w>0):
			ix, iy = x-w/2, y-h/2
			initTracking = True


def multitracker(frame, status,ROIS):
	if status:
		for ROI in ROIS:
			tracker = kcftracker.KCFTracker(True, True, True)
			tracker.init(ROI, frame)
			trackerm.append(tracker)
	else:
		ROIS = []
		for tracker in trackerm:
			ROI = list(map(int, tracker.update(frame)))
			ROIS.append(ROI)
		return ROIS

if __name__ == '__main__':

	rbg = cv2.VideoCapture("./rgb.avi")
	thermal = cv2.VideoCapture("./thermal.avi")
	inteval = 30
	tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
	#if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

	detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
	my_filter = KalmanFilter(dim_x=2, dim_z=1)
	while(thermal.isOpened()):

		boundingbox=[]
		avearray2, Normalizedlist, Normalizedlist2= [], [], []
		avearray = []
		retr, rgbframe = rbg.read()
		ret, thframe = thermal.read()
		if initTracking :
			results = detector.detect_face(rgbframe)

			if results is not None:

				total_boxes = results[0]
				points = results[1]

				for p in points:
					lf = rg = p[2]
					top = bm = p[7]
					for i in range(3, 5):
						if p[i] < lf:
							lf = p[i]
						if p[i] > rg:
							rg = p[i]
						if p[i + 5] > top:
							top = p[i + 5]
						if p[i + 5] < bm:
							bm = p[i + 5]

					ROIS.append([int(lf), int(bm), int(rg-lf), int(top-bm)])
					ROIS.append([int(lf-20), int(bm), int(rg - lf), int(top - bm)])

				multitracker(rgbframe,True,ROIS)
				boundingbox = ROIS[0]
				t1 = time()

				grayframe = cv2.cvtColor(rgbframe, cv2.COLOR_BGR2HSV)
				for xaxis in range(boundingbox[0], boundingbox[0] + boundingbox[2]):
					for yaxis in range(boundingbox[1], boundingbox[1] + boundingbox[3]):
						Normalizedlist.append(grayframe[yaxis][xaxis][0])

				avearray2 = np.asarray(Normalizedlist)
				totalmean = np.mean(avearray2)
				my_filter.x = np.array([[totalmean],
										[0.]])  # initial state (location and velocity)

				my_filter.F = np.array([[1., 1.],
										[0., 1.]])  # state transition matrix

				my_filter.H = np.array([[1., 0.]])  # Measurement function
				my_filter.P *= 1000.  # covariance matrix
				my_filter.R = 5  # state uncertainty
				my_filter.Q = Q_discrete_white_noise(2, .1)  # process uncertainty

				initTracking = False
				onTracking = True
		elif onTracking :

			if not ret:
				break
			t0 = time()
			ROIS = multitracker(thframe, False, ROIS)
			boundingbox = ROIS[0]
			t1 = time()

			cv2.rectangle(thframe, (boundingbox[0], boundingbox[1]),
						  (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)

			grayframe = cv2.cvtColor(thframe, cv2.COLOR_BGR2HSV)
			for xaxis in range(boundingbox[0], boundingbox[0] + boundingbox[2]):
				for yaxis in range(boundingbox[1], boundingbox[1] + boundingbox[3]):
					Normalizedlist.append(grayframe[yaxis][xaxis][0])

			rgbframe = cv2.cvtColor(rgbframe, cv2.COLOR_BGR2HSV)
			for xaxis in range(boundingbox[0], boundingbox[0] + boundingbox[2]):
				for yaxis in range(boundingbox[1], boundingbox[1] + boundingbox[3]):
					Normalizedlist2.append(rgbframe[yaxis][xaxis][0])

			avearray2 = np.asarray(Normalizedlist)
			totalmean = np.mean(avearray2)
			avearray = np.asarray(Normalizedlist2)
			totalmean2 = np.mean(avearray)
			my_filter.predict()
			my_filter.update(totalmean)

			x = my_filter.x
			meanlist.append(totalmean)
			alll = np.asarray(meanlist)
			if len(alll) >= 150 + cutlow:  # fps = 30 frames/second, 300frames = 10 second window (30frames *10seconds)
				HRready = True
				FPS = 30.00
				WINDOW_TIME_SEC = 5
				WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))
				windowStart = len(alll) - WINDOW_SIZE
				window = alll[windowStart: windowStart + WINDOW_SIZE]
				window = np.asarray(window)
				ica = FastICA(whiten=False)
				window = (window - np.mean(window, axis=0)) / np.std(window, axis=0)  # signal normalization
				window = np.reshape(window, (150, 1))
				S = ica.fit_transform(window)  # ICA Part
				fs = 30.0
				lowcut = 0.75
				highcut = 4.0
				detrend = scipy.signal.detrend(S)
				move = running_mean(detrend, 10)

				# plt.plot(move)
				# plt.show()


				y = butter_bandpass_filter(detrend, lowcut, highcut, fs, order=5)
				powerSpec = np.abs(np.fft.fft(y, axis=0)) ** 2
				freqs = np.fft.fftfreq(150, 1.0 / 30)
				MIN_HR_BPM = 50.0
				MAX_HR_BMP = 110.0
				MAX_HR_CHANGE = 2.0
				SEC_PER_MIN = 60
				maxPwrSrc = np.max(powerSpec, axis=1)
				validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
				validPwr = maxPwrSrc[validIdx]
				validFreqs = freqs[validIdx]
				maxPwrIdx = np.argmax(validPwr)
				hr = validFreqs[maxPwrIdx]
				cutlow = cutlow + 5
				out6 = hr * 60
				result.append(out6)
				ave = np.asarray(result)
				out6 = int(np.mean(ave))

				previous = out6
				# lock2.acquire()
				textFile = open('bpmTemp.txt', 'w')
				textFile.write(str(out6))
				# textFile.close()
				# lock2.release()

				tao = str('%.2f' % (out6))

				ce = 'BPM: ' + tao
				cv2.putText(thframe, ce, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))

			cv2.imshow('tracking', thframe)
			c = cv2.waitKey(inteval) & 0xFF
			if c==27 or c==ord('q'):
				break

	plt.plot(alll)
	plt.show()
	rbg.release()
	thermal.release()
	cv2.destroyAllWindows()
