from pynmea import nmea
import serial, time, sys, threading, datetime, shutil

class GPS:

    def __init__(self):

        self.ser = serial.Serial()
        self.ser.baudrate = 4800
        self.coord = 0

        try:

            self.ser.port = '/dev/ttyUSB0'
            self.ser.timeout = 1
            self.ser.open()

        except:

            self.ser.port = '/dev/ttyUSB1'
            self.ser.timeout = 1
            self.ser.open()


        print('GPS loaded.. .warming up')
        time.sleep(2)

    def run_threaded(self):
        return self.coord

    # def run(self):
    #     f = next(self.stream)
    #     frame = f.array

    #     self.rawCapture.truncate(0)
    #     return frame

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            try:
                
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
                    pos_y = lat
                    pos_x = -long       
        
    #               print(pos_y)
    #               print(pos_x)
    #               print(pos_y + ',' + pos_x)
    #               latLong[0] = (pos_y)
    #               latLong[1] = (pos_x)
                    self.coord = (str(pos_y) + ',' + str(pos_x))
                    #print(latLongString)
#                return self.coord

            except:
                pass
 #               return self.coord
                
            #GPS reading code goes



    def shutdown(self):
        # indicate that the thread should be stopped
        print('stoping GPS')
        time.sleep(.5)
