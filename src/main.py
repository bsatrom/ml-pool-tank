import json
import os
import time
from periphery import I2C
import notecard
import sys
import cv2
from edge_impulse_linux.image import ImageImpulseRunner

productUID = "com.blues.bsatrom:pool_tank_monitor"

dir_path = os.path.dirname(os.path.realpath(__file__))
modelfile = os.path.join(dir_path, '../model/pressure-model.eim')
print(f'Using model at {modelfile}')

print("Connecting to Notecard...")
port = I2C("/dev/i2c-1")
card = notecard.OpenI2C(port, 0, 0, debug=True)

def now():
  return round(time.time() * 1000)

def get_webcams():
  port_ids = []
  for port in range(5):
    print("Looking for a camera in port %s:" %port)
    camera = cv2.VideoCapture(port)
    if camera.isOpened():
      ret = camera.read()[0]
      if ret:
        backendName =camera.getBackendName()
        w = camera.get(3)
        h = camera.get(4)
        print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
        port_ids.append(port)
      camera.release()
  return port_ids

def main():
  print(f'Configuring Product: {productUID}...')

  req = {"req": "hub.set"}
  req["product"] = productUID
  req["mode"] = "periodic"
  req["outbound"] = 60
  req["inbound"] = 120
  req["align"] = True

  card.Transaction(req)

main()

while True:
  print("Taking a sample from the camera...")

  with ImageImpulseRunner(modelfile) as runner:
    try:
      model_info = runner.init()
      print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
      labels = model_info['model_parameters']['labels']

      #port_ids = get_webcams()
      #if len(port_ids) == 0:
      #  raise Exception('Cannot find any webcams')
      #if len(port_ids)> 1:
      #  raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
      #videoCaptureDeviceId = int(port_ids[0])
      videoCaptureDeviceId = 0

      camera = cv2.VideoCapture(videoCaptureDeviceId)
      ret = camera.read()[0]

      if ret:
        backendName = camera.getBackendName()
        w = camera.get(3)
        h = camera.get(4)
        print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
        camera.release()
      else:
        raise Exception("Couldn't initialize selected camera.")

      next_frame = 0 # limit to ~10 fps here
      inference_count = 0

      for res, img in runner.classifier(videoCaptureDeviceId):
        if (next_frame > now()):
          time.sleep((next_frame - now()) / 1000)

        next_frame = now() + 500
        inference_count += 1

        if inference_count == 10:
          high_score = 0
          # print('classification runner response', res)

          if "classification" in res["result"].keys():
            req = {"req": "note.add"}
            req["sync"] = True

            noteBody = {"inference_time": res['timing']['dsp'] + res['timing']['classification']}
            print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
            print('', flush=True)

            sorted_items = sorted(res['result']['classification'].items(), key=lambda x:x[1], reverse=True)
            inferred_state = sorted_items[0][0]
            noteBody["tank-state"] = inferred_state
            noteBody["classification"] = res['result']['classification']
            req["body"] = noteBody

            card.Transaction(req)

            # If the state is low or high, send a different Note with an
            # alert message
            req = {"req": "note.add"}
            req["sync"] = True
            req["file"] = "tank-alert.qo"
            if inferred_state == 'tank-pressure-low':
              req["body"] = {"message": "Tank pressure is low. Clean impeller."}
              card.Transaction(req)
            elif inferred_state == 'tank-pressure-high':
              req["body"] = {"message": "Tank pressure is high. Backwash filter."}
              card.Transaction(req)
          break
    finally:
      if (runner):
        runner.stop()

  print("Pausing until next capture...")
  time.sleep(240)
