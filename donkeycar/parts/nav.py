import googlemaps
#from direction_consts import API_KEY, TRAVEL_MODE
from datetime import datetime as dt
from pprint import pprint
import serial
import time
import sys
import threading
import shutil
import math
from pynmea import nmea
#directions = googlemaps.directions
#gmap = googlemaps.Client(key=API_KEY)

#destination_address=input('Enter Destination: ')
#https://github.com/googlemaps/google-maps-services-python PYTHON GOOGLEMAPS

#39.680, -75.751  THIS IS THE LOCATION OF EVANS HALL
#39.679640, -75.751837 Is the location of Brown lab
#39.681074, -75.752658 Is the location of sharp lab

list1 = []
waypoints = ''
defaultWaypoints = 0

def askUserForWaypoints(waypoints_list, moreThanOneWaypoint):
	###right now, the function just takes the user to the waypoint
	###does not continue to the end address
	userInput = input('Enter y for more waypoints, n for no more waypoints: ')
	if(moreThanOneWaypoint == 1):
		waypoints_list += '|'
	if(moreThanOneWaypoint == 21):
		print('\nReached maximum waypoint limit. Leaving function.')
		return waypoints_list
	if(userInput == 'y'):
		lat = input('Enter latitude of waypoint: ')
		waypoints_list += str(lat)
		long = input('Enter longitude of waypoint: ')
		waypoints_list += (',' + str(long))
		moreThanOneWaypoint += 1
		return(askUserForWaypoints(waypoints_list, defaultWaypoints))
	elif(userInput == 'n'):
		return waypoints_list
	else:
		print('\nUnrecognized input. Try again.')
		return(askUserForWaypoints(waypoints_list, defaultWaypoints))



class Navigator:
	def __init__(self):

		now = dt.now()
		starting_address = 39.679640,-75.751837#Brown Lab
		#starting_address = getLatLong()
		ending_address = 39.681074,-75.752658#Sharp Lab
		#waypoint_intersections = askUserForWaypoints(waypoints, defaultWaypoints)
		#directions_result = gmap.directions(starting_address,
                #                    ending_address,
                #                    mode=TRAVEL_MODE,
                #                    waypoints=waypoint_intersections)
		#results_dict = directions_result[0]
		#steps = results_dict['legs'][0]['steps']

		self.list1TXT = []

		self.list1 = []
		#self.directions = steps
		self.waypoints = ''
		self.defaultWaypoints = 0

		#turnDirections!!!
		#	"g" = straight,	  "r" = left,	"y" = right
		self.turnDirection = "g"
		
		
		self.ser = serial.Serial()
		self.ser.baudrate = 4800
		self.lat = 0
		self.long = 0
		
		try:
			self.ser.port = '/dev/ttyUSB0'
			self.ser.timeout = 1
			self.ser.open()

		except:
			self.ser.port = '/dev/ttyUSB1'
			self.ser.timeout = 1
			self.ser.open()

		print('Maps Navigation Loaded.. . warming up and writing files')
		time.sleep(2)

		#self.list1 = self.getIntersectionDirections()
		directionstxt = open( 'directions.py', 'w')
		for elem in self.list1:
			directionstxt.write(str(elem) +'\n')
		directionstxt.close()

		with open("short_figure_8.txt") as directionslist:
			for line in directionslist:
				print(line)
				if (line[1].isdigit()):
					self.list1TXT.append(float(line[:-2]))
				elif "left" in line:
					self.list1TXT.append("left")
				elif "right" in line:
					self.list1TXT.append("right")
				elif "straight" in line:
					self.list1TXT.append("straight")

		
	def run_threaded(self):
		return self.turnDirection

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			try:

	
#				if(self.distanceTolerance() <= .000045):
#					if (list1 == 'straight'):
#						self.turnDirection = "g"
#					elif (list1 == 'right'):
#						self.turnDirection = "y"
#					elif (list1 == 'left'):
#						self.turnDirection = "r"
#					self.updateDirections()
				if(self.distanceToleranceTXT() <= .000045):
					if (list1TXT == 'straight'):
						self.turnDirection = "g"
					elif (list1TXT == 'right'):
						self.turnDirection = "y"
					elif (list1TXT == 'left'):
						self.turnDirection = "r"
					self.updateDirections()
			except:
				pass
	
	def updateLatLong(self):
		data = self.ser.readline()
		data = data.decode("utf-8")#converts data from bytes to string for parsing
                
		if (data[0:6] == '$GPGGA'):
			gpgga = nmea.GPGGA()
			gpgga.parse(data)
			lats = gpgga.latitude
			longs = gpgga.longitude
                
                    #convert degrees, decimal minutes to decimal degrees 
			lat1 = (float(lats[2]+lats[3]+lats[4]+lats[5]+lats[6]+lats[7]+lats[8]))/60
			lat = (float(lats[0]+lats[1])+lat1)
			long1 = (float(longs[3]+longs[4]+longs[5]+longs[6]+longs[7]+longs[8]+longs[9]))/60
			long = (float(longs[0]+longs[1]+longs[2])+long1)            

                    #calc position
			self.lat = lat
			self.long = -long    
			self.coord = (str(self.lat) + ',' + str(self.long))


	def distanceTolerance(self):
		#returns if the current lat long is within 15 ft tolerance of actual intersection
		theoretical = math.sqrt(self.list1[0] ** 2 + self.list1[1] ** 2)
		#actual = math.sqrt(self.lat**2 + self.long**2)
		tolerance = theroetical #- actual
		return tolerance

	def distanceToleranceTXT(self):
		#returns if the current lat long is within 15 ft tolerance of actual intersection
		theoretical = math.sqrt(self.list1TXT[0] ** 2 + self.list1TXT[1] ** 2)
		actual = math.sqrt(self.lat**2 + self.long**2)
		tolerance = abs(theroetical - actual)
		return tolerance

	def directionsOntheGo(self):
		now = dt.now()
		starting_address = 39.679640,-75.751837#Brown Lab
		#starting_address = getLatLong()
		ending_address = 39.681074,-75.752658#Sharp Lab
		#ending_address = read from memory
		directions_result = gmap.directions(starting_address,
                                    ending_address,
                                    mode=TRAVEL_MODE)
		results_dict = directions_result[0]
		steps = results_dict['legs'][0]['steps']
		self.directions = ''
		self.directions = getIntersectionDirections(steps)
		return self.directions

		
	def getIntersectionDirections(self):
		for step in self.directions:
			for value in step['start_location']:
				self.list1.append(step['start_location'][value])
			if 'maneuver' in step:
				self.list1.append(step['maneuver'])
				if(step['maneuver'] == 'straight'):
					print('hi')
			else:
				self.list1.append('straight')
		if(self.list1[0] < 0):
			self.flipLatLong(self.list1)
		return self.list1
	
	def latLongtoString(list):
		#returns the lat long as a string for the googlemaps https request
		return (str(list[0]) + ',' + str(list[1]))

	def updateDirections(self):
		#When an intresection is reached, remove the lat long and turn direction from the list
		self.list1.pop(0)
		self.list1.pop(0)
		self.list1.pop(0)
		return self.list1

	def flipLatLong(self, list):
		#Flips the list only if the longitude is returned before the latitude
		i = 0
		#print('flipped\n')
		while(i < len(list)):
			b = list[i]
			list[i] = list[i + 1]
			list[i + 1] = b
			b = 0
			i += 3
		return list

	def shutdown(self):
		#indicate that the thread should be stopped
		print('Stopping Maps Navigation')
		time.sleep(.5)

