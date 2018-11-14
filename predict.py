from mvnc import mvncapi as mvnc
import cv2
import argparse
import time
import threading
import numpy as np

#mode = 'image'
image = 'images/Traffic.jpg'
mode = 'video'
video = 'videos/uba.mp4'
video_mode = True if mode == "video" else False

args = dict(mode=mode, num=1, image=image, video=video)

#Intel's Neural Compute Stick
mvnc.global_set_option( mvnc.GlobalOption.RW_LOG_LEVEL, 2 )
devices = mvnc.enumerate_devices()
if len(devices) == 0:
	print( "No devices found..." )
	quit()


with open( './model/graph', mode='rb' ) as f:
	graphfile = f.read()
	graph = mvnc.Graph( 'graph' )
	

class feed_forward_thread( threading.Thread ):
	
	classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
	colors = {
		"aeroplane": (254.0, 254.0, 254), 
		"bicycle": (239.9, 211.7, 127), 
		"bird": (225.8, 169.3, 0), 
		"boat": (211.7, 127.0, 254), 
		"bottle": (197.6, 84.7, 127), 
		"bus": (183.4, 42.3, 0),
		"car": (255, 0, 0), 
		"cat": (155.2, -42.3, 127), 
		"chair": (141.1, -84.7, 0), 
		"cow": (127.0, 254.0, 254), 
		"diningtable": (112.9, 211.7, 127), 
		"dog": (98.8, 169.3, 0), 
		"horse": (84.7, 127.0, 254), 
		"motorbike": (70.6, 84.7, 127), 
		"person": (56.4, 42.3, 0), 
		"pottedplant": (42.3, 0, 254), 
		"sheep": (28.2, -42.3, 127), 
		"sofa": (14.1, -84.7, 0), 
		"train": (0, 254, 254), 
		"tvmonitor": (-14.1, 211.7, 127)
	}

	def __init__( self, device, args, graph, video=False ):
		threading.Thread.__init__( self )
		self.device = None
		self.fifoIn = None
		self.fifoOut = None
		self.video_mode = video
		self.args = args
		self.actual_width = 0.0
		self.actual_height = 0.0
		self.input_height = 416
		self.input_width = 416
		self.video = False
		self.graph = graph
		self.open_device_load_graph( device )

	def open_device_load_graph( self, device ):
		self.device = mvnc.Device( device )
		self.device.open()
		self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos( self.device, graphfile )
		
	def preproces_image(self, input):
		#self.actual_height, self.actual_width, _ = input.shape
		resized_image = cv2.resize(input, (self.input_height, self.input_width), interpolation = cv2.INTER_CUBIC)
		image_data = np.array(resized_image, dtype='f')
		image_data /= 255.
		#image_array = np.expand_dims(image_data, 0)

		return image_data


	def sigmoid(self, x):
		return 1. / (1. + np.exp(-x))


	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		out = e_x / e_x.sum()
		return out


	def iou(self, boxA, boxB):
		# boxA = boxB = [x1,y1,x2,y2]

		# Determine the coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
 
		# Compute the area of intersection
		intersection_area = (xB - xA + 1) * (yB - yA + 1)
 
		# Compute the area of both rectangles
		boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
		# Compute the IOU
		iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

		return iou
		
		
	def run_interference(self, image):
		image = self.preproces_image(image)
		self.graph.queue_inference_with_fifo_elem(self.fifoIn, self.fifoOut, image, 'user object')
		prediction, _ = self.fifoOut.read_elem()
		out = self.postprocess(prediction, image, 0.3, 0.3 )
		
		return out


	def postprocess(self, predictions, input_image, score_threshold, iou_threshold):

		# input_image = cv2.resize(input, (self.input_height, self.input_width), interpolation = cv2.INTER_CUBIC)

		anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
		thresholded_predictions = []
		predictions = np.reshape(predictions, (13, 13, 5, 25))

		# IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
		for row in range(13):
			for col in range(13):
				for b in range(5):
					tx, ty, tw, th, tc = predictions[row, col, b, :5]
					center_x = (float(col) + self.sigmoid(tx)) * 32.0
					center_y = (float(row) + self.sigmoid(ty)) * 32.0

					roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
					roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

					final_confidence = self.sigmoid(tc)

					# Find best class
					class_predictions = predictions[row, col, b, 5:]
					class_predictions = self.softmax(class_predictions)
					class_predictions = tuple(class_predictions)
					best_class = class_predictions.index(max(class_predictions))
					best_class_score = class_predictions[best_class]

					# Compute the final coordinates on both axes
					left   = int(center_x - (roi_w/2.))
					right  = int(center_x + (roi_w/2.))
					top    = int(center_y - (roi_h/2.))
					bottom = int(center_y + (roi_h/2.))
		
					if( (final_confidence * best_class_score) > score_threshold):
						thresholded_predictions.append([ [left,top,right,bottom], final_confidence * best_class_score, self.classes[best_class] ])

		nms_predictions = []
		if len(thresholded_predictions) != 0:
			thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)
			nms_predictions = self.non_maximal_suppression(thresholded_predictions, iou_threshold)

			# Draw boxes with texts
			for i in range(len(nms_predictions)):
				color = self.colors[nms_predictions[i][2]]
				best_class_name = nms_predictions[i][2]
				textWidth = cv2.getTextSize(best_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0] + 10
			
				input_image = cv2.rectangle(input_image, (nms_predictions[i][0][0], nms_predictions[i][0][1]), (nms_predictions[i][0][2],nms_predictions[i][0][3]), color)
				input_image = cv2.rectangle(input_image, (nms_predictions[i][0][0], nms_predictions[i][0][1]), (nms_predictions[i][0][0]+textWidth, nms_predictions[i][0][1]+20), color, -1)
				input_image = cv2.putText(input_image, best_class_name, (nms_predictions[i][0][0]+5, nms_predictions[i][0][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4)

		#resize image back to original and show it
		# input_image = cv2.resize(input_image,(self.actual_width, self.actual_height))

		return input_image, nms_predictions


	def non_maximal_suppression(self, thresholded_predictions, iou_threshold):

		nms_predictions = []
		# Add the best B-Box because it will never be deleted
		nms_predictions.append(thresholded_predictions[0])
		# For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
		# thresholded_predictions[i][0] = [x1,y1,x2,y2]
		i = 1
		while i < len(thresholded_predictions):
			n_boxes_to_check = len(nms_predictions)
			to_delete = False

			j = 0
			while j < n_boxes_to_check:
				curr_iou = self.iou(thresholded_predictions[i][0],nms_predictions[j][0])
				if(curr_iou > iou_threshold ):
					to_delete = True

				j = j+1

			if to_delete == False:
				nms_predictions.append(thresholded_predictions[i])
			i = i+1

		return nms_predictions

	def run( self ):
		if self.video_mode:
			fps = 0.0

			#Webcam mode, else video file mode
			#if self.args["video"].isdigit():
			#	self.args["video"] = int( self.args["video"]) 

			cap = cv2.VideoCapture(self.args["video"])
			cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
			cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)

			while True:
				start = time.time()
				ret, display_image = cap.read()

				if not ret: 
					print( "No image found from source, exiting" )
					break

				output_image, boxes = self.run_interference(display_image)

				fps  = ( fps + ( 1 / (time.time() - start) ) ) / 2
				output_image = cv2.putText(output_image, "fps: {:.1f}".format(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4 )
				cv2.imshow("Press q to quit", output_image)
	
				if cv2.getWindowProperty( "Press q to quit", cv2.WND_PROP_ASPECT_RATIO ) < 0.0:
					print( "Window closed" )
					break
				elif cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
					print( "Q pressed" )
					break


			cap.release()
			cv2.destroyAllWindows()

		else:
			start = time.time()
			image = cv2.imread( self.args["image"] )
			output_image, boxes = self.run_interference(image)

			print( "Time took: {:.1f} sec".format(time.time() - start) )

			cv2.imshow("Press q to quit", output_image)
			cv2.waitKey(0)

		#Close device and with that the thread
		self.graph.destroy()
		self.fifoIn.destroy()
		self.fifoOut.destroy()
		self.device.close()


#Run script
threads = []
threads.append(feed_forward_thread(devices[0], args, graph, video=video_mode)) 

#run thread
for thread in threads:
	thread.start()

#wait until threads are done
for thread in threads:
	thread.join()

#Done!!
print('Finished')
