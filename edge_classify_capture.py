"""A demo to classify Raspberry Pi camera stream."""

import argparse
import platform
import os
import io
import time
from collections import deque
import numpy as np
import picamera
import cv2
from PIL import Image
from PIL import ImageDraw
from abeja.datalake import Client as DatalakeClient

import edgetpu.detection.engine
import edgetpu.classification.engine
import requests
import json
import urllib.parse

# please input the line bot address
line_to = 'xxxx'
line_post_url = 'https://api.line.me/v2/bot/message/push'
# please input the line access key
line_auth_key = 'xxxx'

bitly_post_url = 'https://api-ssl.bitly.com/v4/shorten'
# please input the bitly access key
bitly_auth_key = 'xxxx'

# please input the ABEJA Platform account information
abeja_organization_id = 'xxxx'
abeja_channel_id = 'xxxx'
abeja_user_id = 'xxxx'
abeja_personal_access_token = 'xxxx'

def get_bitly_url(download_url):
  headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}'.format(bitly_auth_key)
  }
  body = {
    'long_url': download_url
  }
  res = requests.post(bitly_post_url,
          json.dumps(body),
          headers=headers
        )
  return 'https://{}'.format(res.json()['id'])

def upload_image_datalake(image_path):
  abeja_credential = {
    'user_id': abeja_user_id,
    'personal_access_token': abeja_personal_access_token
  }
  datalake_client = DatalakeClient(organization_id=abeja_organization_id, credential=abeja_credential)
  channel = datalake_client.get_channel(abeja_channel_id)
  res = channel.upload_file(image_path)
  datalake_file = channel.get_file(file_id=res.file_id)
  content = datalake_file.get_file_info()
  return content['download_url']

def push_line_message(image_url):
  headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}'.format(line_auth_key)
  }
  body = {
    'to' : line_to,
    'messages':[
      {
        'text': '【緊急通知】怪しいやーつがご自宅の庭に侵入しています',
        'type': 'text'
      },
      {
        'originalContentUrl' : image_url,
        'previewImageUrl' : image_url,
        'type': 'image'
      }
    ]
  }
  res = requests.post(line_post_url,
          json.dumps(body),
          headers=headers
       )

def main():
  default_model_dir = '../all_models'
  default_model = 'mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite'
  default_labels = 'coco_labels.txt'
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='.tflite model path',
                      default=os.path.join(default_model_dir,default_model))
  parser.add_argument('--labels', help='label file path',
                      default=os.path.join(default_model_dir, default_labels))
  args = parser.parse_args()

  with open(args.labels, 'r') as f:
    pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
    labels = dict((int(k), v) for k, v in pairs)

  engine = edgetpu.detection.engine.DetectionEngine(args.model)

  with picamera.PiCamera() as camera:
     camera.resolution = (640, 480)
     camera.framerate = 30
     camera.annotate_text_size = 20
     _, width, height, channels = engine.get_input_tensor_shape()
     camera.start_preview()
     try:
       stream = io.BytesIO()
       fps = deque(maxlen=20)
       fps.append(time.time())
       for foo in camera.capture_continuous(stream,
                                              #format='rgb',
                                              format='jpeg',
                                              use_video_port=True,
                                              resize=(width, height)):
         stream.truncate()
         stream.seek(0)
         # input = np.fromstring(stream.getvalue(),dtype=np.uint8)
         camera.capture(stream, format='jpeg')
         img = Image.open(stream)
         draw = ImageDraw.Draw(img)
         output_name = 'images/test_{}.jpeg'.format(time.time())
         # Run inference.
         ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True, relative_coord=False, top_k=3)

         # Display result.
         if ans:
           for obj in ans:
             print ('-----------------------------------------')
             if labels:
               print(labels[obj.label_id])
             print ('score = ', obj.score)
             box = obj.bounding_box.flatten().tolist()
             print ('box = ', box)
             # Draw a rectangle.
             draw.rectangle(box, outline='blue')
             draw.text((box[0],box[1]), '{} {}'.format(labels[obj.label_id], str(obj.score)), 'red')
             if labels[obj.label_id] == 'person' and obj.score > 0.6:
               img.save(output_name)
               url = upload_image_datalake(output_name)
               short_url = get_bitly_url(url)
               push_line_message(short_url)
               exit()
           img.save(output_name)
           if platform.machine() == 'x86_64':
             # For gLinux, simply show the image.
             img.show()
           elif platform.machine() == 'armv7l':
             # For Raspberry Pi, you need to install 'feh' to display image.
             subprocess.Popen(['feh', output_name])
           else:
             print ('Please check ', output_name)
         else:
           print ('No object detected!')

     finally:
         camera.stop_preview()


if __name__ == '__main__':
    main()
