
# Example usage of UKRegRead python module
# SJM/MCL 2018  www.marvellconsultants.com

# Note that the UKRegRead module requires Python 3.x with PILlow, openCV (cv2) and NumPy extensions
# Plus UKNumberPlate.ttf true-type font file available from:
#  https://www.dafont.com/uk-number-plate.font
# Plus UKPlates66x20P.xml and UKChars33_16x25_11W.xml haar-cascade xml files
# All 3 support files present in the working directory.

# Specify an image file-name (with optional wild-cards) on the command line, for example:
#   python Example.py samples/*.jpg
# Example.py will attempt a plate-read on each file and report the results, finally displaying
# an annotated thumbnail of the last image.
# See documentation for features, limitations etc


import cv2, os, glob, timeit, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import UKRegOCR

# target_size:
#  Tuple TgtWH should aproximate the average, mid-field size in pixels (W,H) of a UK number plate
#  as seen by by your camera.  The plate should be roughly horizontal and reasonably un-distorted,
#  the reader can cope with no more than 10 degrees of rotation and skew.  Pre-process your images
#  with one of openCV's afine transforms if necessary.
#  Note that a plate height of 50 pixels corresponds to a character height of about 33 pixels, the reader
#  works well down to a character-height lower limit of around 25 pixels, less well below that.
#
TgtWH = (170,50)

# -------------------------------------------------------------------------

# Feed image file(s) into the ReadPlate engine, print results and monitor read-times.
# display an annotated thumbnail of final image.

f = 'samples/*.jpg'
# check command-line arguments...
if len(sys.argv) > 1:
  f = sys.argv[1]
else:
  print('Please enter an image file name on the command line (may contain wildcards).')
  print('Defaulting to', f, '...')
images = glob.glob(f)
Colrs = {'Y':'Yellow', 'B':'Black', 'W':'White', '-':'DontKnow'}
UKFont = ImageFont.truetype('UKNumberPlate.ttf', 24) # used for annotation
Icnt = 0
Mcnt = 0
Ttot = 0
for fn in images:
  Icnt += 1
  fn = fn.replace('\\', '/') # for MSW compatibility 
  if not os.path.isfile(fn): continue
  in_img = cv2.imread(fn)
  print(Icnt, '/', len(images), ' - ', fn, ':-')
  t0 = timeit.default_timer()
  result = UKRegOCR.ReadPlate(in_img, TgtWH) # can return either error_string or reg,conf,bbox,plate_colour
  t1 = timeit.default_timer()
  Ttot += t1 - t0
  if type(result) is tuple:
    regn,cnf,bbox,pcol = result
    if pcol in Colrs: pcol = Colrs[pcol]
    print('Reg:', regn, '  conf:', cnf, '  b/box:', bbox, '  Plate_Colour:', pcol, '  in {:0.1f}s'.format(t1-t0))
    if len(images) == Icnt:
      # create an annotated output thumbnail
      asp = in_img.shape[1]/in_img.shape[0] # w/h
      rgb_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
      _,_,iw,ih = bbox
      p_crop = Image.fromarray(rgb_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])
      p_img = Image.fromarray(rgb_img)
      dh = 288
      scale = dh / in_img.shape[0]
      dw = int(dh * asp)
      # re-size to height=dh maintaining aspect ratio
      p_img.thumbnail((dw,dh))
      bbox = bbox[0]*scale, bbox[1]*scale, bbox[2]*scale, bbox[3]*scale
      # indicate plate AOI with red rectangle
      ImageDraw.Draw(p_img).rectangle((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]), None, (255,0,0))
      # show plate as-read w/ correct f/g & b/g colours
      bcol = (255,255,255) # white b/g
      fcol = (0,0,0) # black f/g
      if pcol[0] == 'B': bcol,fcol = fcol,bcol # white-on-black
      elif pcol[0] == 'Y': bcol = (255,255,0) # yel b/g
      ph = UKFont.getsize('X')[1] * 1.33
      cw = UKFont.getsize(regn)[0]
      pw = ph * 5
      ImageDraw.Draw(p_img).rectangle((dw-pw, 0, dw-1, ph), bcol, None)
      ImageDraw.Draw(p_img).text((dw-cw-(pw-cw)/2, 2), regn, fcol, font=UKFont)
      # add original AOI 1:1 w/ white border
      p_img.paste(p_crop, (dw-iw-1, dh-ih-1))
      ImageDraw.Draw(p_img).rectangle((dw-iw-2, dh-ih-2, dw-1, dh-1), None, (255,255,255))
      p_img.show()
  else:
    print('Read failure:', result)
  print('')  
# print summary
print('processed', Icnt, 'images in {:0.1f}s,'.format(Ttot), 'average read-time: {:0.2f}s'.format(Ttot/max(1, Icnt)))
