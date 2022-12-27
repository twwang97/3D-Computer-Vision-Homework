############################
#                          #
# Time Recording           #
#                          #
# Author: David Wang       #
# Created on Oct. 24, 2022 #
#                          #
############################


import time

class timeRecording_:
	def __init__(self):
		self.start = time.time()
		self.end = time.time()
		self.elapsed = time.time()

	def record(self):
		self.end = time.time()
		self.elapsed = self.end - self.start
		print("\nelapsed time: {:.2f} seconds.\n".format(self.elapsed))