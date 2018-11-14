import tensorflow as tf
import NeuralNetwork
import os
import cv2
import time
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--image', help='foo help')
#args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image = 'Traffic.jpg'
cv2_image = cv2.imread("./images/{}".format(image))

NN = NeuralNetwork.Net(Debugging=True) # 580 ms
image = NN.image
picture = NN.preproces_image(cv2_image) # nothing

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	prediction = sess.run(NN.predict(), feed_dict={image: picture}) # 200 ms

	# Save Network
	saver = tf.train.Saver()
	saver.save(sess, "./model/NN.ckpt")
	tf.train.write_graph( sess.graph_def, "./model/", "NN.pb", as_text=False )

	output_image, boxes = NN.postprocess(prediction, cv2_image, 0.3, 0.3) # 32 ms
	cv2.imshow('image', output_image)
	cv2.waitKey(0)
