""""Demo for use yolo v2"""
import tensorflow as tf
import NeuralNetwork
import os
import cv2
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='foo help')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NN = NeuralNetwork.Net(Debugging=True)
image = NN.image


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	video_path = "videos/uba.mp4"
	camera = cv2.VideoCapture(video_path)
	cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

	# Prepare for saving the detected video
	sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#	fourcc = cv2.VideoWriter_fourcc(*'mpeg')
#	vout = cv2.VideoWriter()
#	vout.open("videos/ubaout.mp4", fourcc, 20, sz, True)
	while True:
		res, frame = camera.read()
		if not res:
			break
		start = time.time()
		picture = NN.preproces_image(frame)
		prediction = sess.run(NN.predict(), feed_dict={image:picture})
		output_image, boxes = NN.postprocess(prediction, frame, 0.3, 0.3)
		print("Time took to compute: " + str((time.time()-start)*1000) + "ms")
		cv2.imshow("detection", output_image)
		# Save the video frame by frame
#		vout.write(image)
		if cv2.waitKey(110) & 0xff == 27:
			break
#		vout.release()
#		camera.release()

	#Save Network
#	saver = tf.train.Saver()
#	saver.save(sess, "./model/NN.ckpt")
#	tf.train.write_graph( sess.graph_def, "./model/", "NN.pb", as_text=False )
