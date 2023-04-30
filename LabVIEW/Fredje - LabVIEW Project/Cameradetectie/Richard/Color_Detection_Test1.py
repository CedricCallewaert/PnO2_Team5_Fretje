status = 5

def start_stream():
	global status
	status = 10

def get_frame():
	global status
	status += 1
	

def close_stream():
	global status
	status = 0

def read_status():
	global status
	return status
