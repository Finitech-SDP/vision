# UKRegOCR.py
# SJM / MCL   2018   www.marvellconsultants.com
#
# Import this as a module for access to:
#   ReadPlate(img, target_size, plateXYWH)
#   lookForPlate(img, target_size) and
#   WhtOrYel(img, AOI)
# To run as a script, please provide an image file name on the command line, results will be printed to stdout
# See PDF documentaion for mode details

# Dependencies:
#   Python 3.x with openCV (cv2) & numpy librarys
#   Plus:
#     OCRFonts module (OCRFonts.py)
#     UKPlates66x20P.xml and
#     UKChars33_16x25_11W.xml haar-cascade classifier files
#     - All in the same directory as UKRegOCR.py.

# target_size:
#  Tuple TgtWH should approximate the nominal, mid-field size in pixels (W,H) of a UK number plate
#  as seen by by your camera.  The plate should be roughly horizontal and reasonably un-distorted,
#  the reader can handle only ~10 degrees max of rotation and skew.
#  Pre-process your images with one of openCV's afine transforms if necessary.
#  Note that a plate height of 50 pixels corresponds to a character height of about 33 pixels, the reader
#  works ok down to a character-height of around 25 pixels in good quality, sharp images, less well below that.
#  If characters are much bigger then processing speed will start to suffer.

# Change Log:
#  01/10/18   First public release

# To Do:
#   Leading zero disqualification in all but @@##@@@ format plates
#   Use logging module for debug o/ps
#   Re-visit initial crop after intial character-classifier detection?
#   Re-visit rotation after initial XC?

import math, time, pprint
import cv2
import numpy as np
import OCRFonts

# Module globals
DEBUG    = False     # set True to print verbose debug info
ASPECT   = 0.625     # nominal aspect ratio of non-I characters as seen in centre-field, width/height
ASPI     = 0.23      # relative width of the I char cf all other chars
PH2TH    = 1.57      # approx text height cf lookForPlate()-detected plate height (varies 1.2 .. 1.8)
PW2CH    = 6         # approx UKPlate text-span-width from observed naked character height
STW2TW   = 8.3       # approx UKPlate string width from naked-text-width, use this when we have a better handle on tw
CAS_OVER = 0.91      # correction for CharCas's over-estimate of char height (about 10%)

PLCAS = cv2.CascadeClassifier('UKPlates66x20P.xml')       # SJM's default cascade plate classifier P=conservative
CHCAS = cv2.CascadeClassifier('UKChars33_16x25_11W.xml')  # SJM's favorite character spotter

# Templates for all valid UK plate formats, with corresponding arbitrary likelyhood scores.
# Scores should be in the range 0..25. # for a number, @ for an alpha. lowest scores first.
PlateLUT = {'#@':0,'@#':0,
            '##@':1,'@##':1,'#@@':1,'@@#':1,
            '@###':2,'###@':2,
            '@@##':3,'##@@':3,
            '@@@#':4,
            '####@':5,'@####':5,
            '@@###':6,'###@@':6,
            '@@@##':7,
            '@@@#@':9,'@#@@@':9,'@@####':9,'####@@':9,
            '@@@###':11,
            '@@@##@':13,'@##@@@':13,
            '@@@####':17,
            '@@@###@':20,'@###@@@':20,
            '@@##@@@':24}

# ----------------- public functions ----------------------

# Look for a target number-plate in the image using HaarCascade classifier
# Return object geometry as (X,Y,W,H) tuple if found, else return (0,0,0,0).
# Use an itterative approach with varying sensitivity parameters to try to obtain exactly one hit
# If that's not possible return just the first hit.
def lookForPlate(img, TargetWH=(170,50), xml=None):
  pltCas = PLCAS
  if xml is not None:
    try: # attempt top load the specified cascade xml file
      pltCas = cv2.CascadeClassifier(xml)
    except:
      return [0,0,0,0] # no xml file
  # Start working at the most sensitive level (z=0), if there's no recognition here then we're done.
  z = 0  # sensitivity factor; must start at zero (most sensitive) for no-hit logic to work
  ll, ul = 0, 99
  pls0 = [0,0,0,0]
  while True:
    Tmin = (int(TargetWH[0] * 0.5), int(TargetWH[1] * 0.5))
    Tmax = (Tmin[0] * 3, Tmin[1] * 3)
    pls = pltCas.detectMultiScale(img, 1.1+(z/300), z+6, 0, Tmin, Tmax)
    c = len(pls)
    if c == 1: return pls[0] # one plate, job done.
    if c == 0:
      if z == 0: return pls0 # no plate
      ul = z
      z = (ll + z)//2
      if z == ll: break # out of options
    else:
      pls0 = pls[0] # keep this option up our sleave
      ll = z
      z = (z + ul)//2
      if z == ll: break # out of options
  # >1 result, carry on slowly increasing Z without limit until 1 or no results
  while True:
    z = max(z+1, z*1.05)
    pls = pltCas.detectMultiScale(img, 1.1+(z/150), int(z), 0, Tmin, Tmax)
    c = len(pls)
    if c == 1: return pls[0]
    if c == 0: break
  return pls0

# Analyse the average colour within the optionally specified crop window for whiteness or yellowness
# Crop process is tolerant of -ve X / Y and also X > W / Y > H
# Assumes img is a 3-plane BGR colour image
# Warning: contrast stretching may distort plate colour, don't us it prior to calling this.
def WhtOrYel(in_img, crop = (0,0,0,0)):
  # crop?
  if (crop[2] > 0) and (crop[3] > 0):
    H = len(in_img)
    W = len(in_img[0])
    (Y0,Y1,X0,X1) = (crop[1], crop[1] + crop[3], crop[0], crop[0] + crop[2])
    Y0 = min(max(Y0, 0), H-1)
    Y1 = min(max(Y1, 0), H-1)
    X0 = min(max(X0, 0), W-1)
    X1 = min(max(X1, 0), W-1)
    img = in_img[Y0:Y1, X0:X1] # crop
  H = len(img)
  W = len(img[0])
  bgr_planes = cv2.split(img)
  if len(bgr_planes) < 3: return '-' # bad crop/image
  b = r = 0
  Ag = np.median(bgr_planes[1])
  for y in range(0, H):
    for x in range(0, W):
      if bgr_planes[1][y][x] >= Ag:
        b += bgr_planes[0][y][x]
        r += bgr_planes[2][y][x]
  if (r == b): return '-'  # monochrome image
  # colour test
  if r == 0: return '-'	# blank image?
  x = b / r
  # print('{:0.2f}'.format(x), end = '')
  if x < 0.9: return 'Y'
  if x > 0.9: return 'W'
  return '-' # in dead-band

# Master plate-read algorithm ------------------
#
# If you have already run the lookForPlate() function on the image and have an (XYWH) tuple
# you can supply this to UKRegOCR() to skip the initial classifer stage and save some CPU time.
#
#  convert source image to grey-scale if necessary
#  identify approx plate position using haar-cascades
#   return error if no plate
#  crop to plate AOI
#  detect and compensate for white-on-black style plates
#  horizontalise the crop
#  de-skew characters
#  estimate character size
#  OCR (template-match) all non-I characters, with itterative char-size optimisation
#  OCR for I characters
#  reject mis-aligned or badly positioned characters 
#  derive an arbitrary confidence level
#  determine plate b/g colour
#
#  Return ('reg', confidence%, (plate-location), 'plate_colour') or '!error_msg'
#   as: ( string, 0..99, (X,Y,W,H), ['W'|'Y'|'B'|'-'] )
#   or: string
#
def ReadPlate(in_img, TargetWH=(170,50), plateXYWH=None):

  # -------------------------- local misc support functions -----------------------

  # Enlarge W & H of a crop geometry w/ approp adj of X & Y, width*2 & height*1.66
  def enlargeCrop(xywh):
    W = int(xywh[2] * 2)
    H = int(xywh[3] * 1.66)
    X = xywh[0] - (W - xywh[2])//2
    Y = xywh[1] - (H - xywh[3])//2
    if X < 0: X = 0
    if Y < 0: Y = 0
    return (X,Y,W,H)

  # Crop an image to the specified XYWH rectangle
  # NOTE if X0,Y0 or X1,Y1 exceed the image boundry then a smaller than expected image will be returned.
  # ASSUMES img is monochrome
  def cropIt(img, crop):
    ih, iw = img.shape
    X0,Y0,X1,Y1 = crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3]
    return img[Y0:Y1, X0:X1]

  # measure the 'morf' image's central block height
  # return the height and a centre-line estimate
  def measureTxH(img, xl, xr):
    xpcs = ()
    for i in range(0,10):
      xpcs += (int(xl + ((xr - xl) * (i + 0.5) / 10)) ,)
    ih = len(img)
    iw = len(img[0])
    yc = n = 0
    Hs = ()
    for xs in xpcs:
      if (xs < 0) or (xs >= iw): continue
      # search up from center-line...
      yt = yb = int(ih/2)
      while yt > 0:
        if img[yt][xs] == 0: break
        yt -= 1
      # search down...
      while yb < ih:
        if img[yb][xs] == 0: break
        yb += 1
      h = yb - yt
      if h > 2:
        Hs += (h, )
        yc += yb + yt
        n += 1
    avH = sjmAvg(Hs)
    if n > 0: ycl = yc / (2 * n)
    else: ycl = ih/3
    return int(avH-0.5), int(ycl)

  # Generate a simple linear equalising LUT from the supplied histogram
  # With hard coded 5% clip limits
  def heq(hist):
    LUT = np.zeros(256)
    bp = 0
    tp = 255
    t = 0
    th = np.sum(hist) * 0.05
    if th >= 1:
      #find bottom 5 percentile point bp
      while True:
        t += hist[bp]
        if t > th: break
        bp += 1
      #find top 5 percentile point
      t = 0
      while True:
        t += hist[tp]
        if t > th: break
        tp -= 1
      # build lut
      r = 255 / (tp - bp)
      for i in range(bp,256): LUT[i] = min(255, int(r * (i - bp)))
    return LUT

  # Calculate a modified average for a list of austensibly +ve numbers by rejecting -ves & any outliers 
  # Delete no more than n/3 outliers, while aiming for a std of < 3
  def sjmAvg(in_lst):
    n = len(in_lst)
    if n == 0: return 0
    lst = in_lst
    s = np.std(lst)
    a = np.average(lst)
    if (s < 3) and (min(lst) >= 0): return a
    while (len(lst) > n * .66) and (len(lst) > 3) and (s > 3):
      newlst = ()
      for i in range(0, len(lst)):
        if (lst[i] >= 0) and (abs(lst[i] - a) < s): newlst += (lst[i],)
      lst = newlst
      if len(lst) > 0:
        s = np.std(lst)
        a = np.average(lst)
    return a

  # Calculate a modified average for a list of numbers by rejecting outliers 
  # Delete no more than n/3 outliers, while aiming for a std of < minSD
  def sjmAvg2(in_lst, minSD):
    n = len(in_lst)
    if n == 0: return 0
    lst = in_lst
    s = np.std(lst)
    a = np.average(lst)
    while (len(lst) > n * .66) and (len(lst) > 3) and (s > minSD):
      newlst = ()
      for i in range(0, len(lst)):
        if abs(lst[i] - a) < s: newlst += (lst[i],)
      lst = newlst
      s = np.std(lst)
      a = np.average(lst)   
    return a

  # If the majority of entries agree, dis-regard all others
  # otherwise just take the average
  def votedAverage(lst):
    av = np.average(lst)
    if len(lst) < 3: return av 
    Bins,Bounds = np.histogram(lst, 20, [-10, 10])
    for i in range(0, len(Bins)):
      if Bins[i] > len(lst) / 2: return Bounds[i]
    return av

  # Nonify outliers from a list (NOT TUPLE) of numbers, by replacing bad numbers with None
  # Edit the supplied list in-place, keep removing out-liers until the Coefficient of Variance
  # of the remaining set falls below the specified maxCV
  # Return number-of-remainers, average, median and CV of remaining set
  # Used by the cascade character spotter
  def rejectOutliers(in_lst, maxCV):
    a = worst = -1
    s = 1
    while True:
      lst = ()
      for i in in_lst:
        if i is not None: lst += (i,)
      s = max(1, np.std(lst))
      a = np.average(lst)
      m = np.median(lst)
      cv = s / max(1, abs(a))
      if cv < maxCV: break
      if len(lst) < 3: break
      #print(a,m,s,cv)
      worst = worstn = None
      for i in range(0, len(in_lst)):
        if in_lst[i] is not None:
          dev = abs(in_lst[i] - a)
          if (worst is None) or (dev > worst):
            worst = dev
            worstn = i
      in_lst[worstn] = None
    return (len(lst), a, m, cv)  

  # Look for UKFont character shapes using SJM's best Haar-cascade classifier
  # Don't return > 7 char locations
  # Reject any dodgey looking shapes with no real contrast
  def seekChrs(img, TxWH):
    Cas = CHCAS
    z = 1
    Tmin = (int(TxWH[0] * 0.5), int(TxWH[1] * 0.5))
    Tmax = (Tmin[0] * 3, Tmin[1] * 3)
    while True:
      z = int(z * 1.2 + 1)
      chs = Cas.detectMultiScale(img, 1.01 + (z/200) , z, 0, Tmin, Tmax)
      if len(chs) < 8: break
    if len(chs) == 0: return [],-1
    # from the chs tuple build a list or rects and a list of areas
    # in prep for rejecting area and centre-line out-liers
    cList = list(range(0,len(chs)))
    chsA = cList.copy()
    for n in range(0, len(chs)):
      ch = chs[n]
      cList[n] = ch
      chsA[n] = ch[2] * ch[3]
    # Reject area out-liers
    N,Aav,Amd,Acv = rejectOutliers(chsA, 0.1) # until Coef of Variance < 0.1
    # reject phantom spots with little or no contrast in the original img
    for n in range(0, len(chsA)):
      if chsA[n] is not None:
        (x,y,w,h) = (cList[n][0],cList[n][1],cList[n][2],cList[n][3])
        chsA[n] = np.std(img[y:y+h, x:x+w])   # we use std dev as a proxy for contrast
    N,av,md,cv = rejectOutliers(chsA, 0.2)
    # Reject Yc out-liers
    for n in range(0, len(chsA)):
      if chsA[n] is not None: chsA[n] = cList[n][1] + (cList[n][3] / 2) # centre y co-ord
    N,Yav,Ymd,Ycv = rejectOutliers(chsA, 0.1)
    # edit cList to reflect current state of chsA 'None's
    for i in range(len(chsA)-1 , -1, -1):
      if chsA[i] is None: cList = np.delete(cList, i, axis=0)
    return cList, Yav

  # Given a map of values find the local peaks over the specified sampling window
  # return a list of (X,Y,val) tuples
  def findLocalPeaks(map, gw):
    result = []
    gx = gw - (1 - (gw % 2)) # force grid width to an odd no.
    gbx2 = int(gx/2)
    H = len(map)
    W = len(map[0])
    Xxcs = np.zeros(W)
    Xys = np.zeros(W)
    Xnzav = np.zeros(W)
    for x in range(0, W):
      n = 0
      for y in range(0, H):
        if map[y,x] > 0:
          Xnzav[x] += map[y,x]
          n += 1
          if map[y,x] > Xxcs[x]:
            Xxcs[x] = map[y,x]
            Xys[x] = y
      if n > 0: Xnzav[x] = (Xnzav[x] - Xxcs[x]) / max(1, n)
      if Xxcs[x] < Xnzav[x] * 1.14: Xxcs[x] = 0
    for x in range(gbx2, W-gbx2):
      tst = Xxcs[x-gbx2:x+gbx2+1].copy()
      mx = np.max(tst)
      if mx < 0.25: continue
      if tst[gbx2] != mx: continue
      # AND UNIQUE PK, all other points must be < mx
      tst[gbx2] = 0
      if mx == np.max(tst): continue
      # and must be sig > average of all other nz points in the central column
      if mx < 1.14 * np.sum(tst) / (1 + cv2.countNonZero(tst)): continue
      result.append((x, int(Xys[x]), mx))
    return result

  # If lh & rh end chars are > maxX apart then cull the weaker one
  # del item(s) in regData accordingly
  # Return true if any deletions made, to allow for itteration
  def delOOBChars(regData, pw, avW, wdthI):
    Xlist = sorted(regData)
    lhi = 0
    rhi = len(regData) - 1
    if rhi <= lhi: return False
    xl,xr = Xlist[lhi], Xlist[rhi]
    wl = avW/2
    if regData[xl][0] == 'I': wl = wdthI/2
    wr = avW/2
    if regData[xr][0] == 'I': wr = wdthI/2
    if (xr-wr) - (xl+wl) <= pw: return False
    if regData[xr][2] > regData[xl][2]:
      #print('ODel', xl, regData[xl])
      del(regData[xl])
      return True
    if regData[xr][2] < regData[xl][2]:
      #print('ODel', xr, regData[xr])
      del(regData[xr])
      return True
    del(regData[xl])
    del(regData[xr])
    return True

  # If the weakest XC in the list is < thr * average of the rest delete it.
  # Return true if any regData item has been removed, to allow for itteration
  # We first build a weakness score based on: low xc, poor Y alignment and
  # excessive inter-char gaps at the ends of the string
  def rejectWeaklings(regData, thr):
    if len(regData) < 3: return False
    Xlist = sorted(regData)
    gaplst = ()
    XClist = []
    prevX = None
    for x in Xlist:
      XClist += [regData[x][2]]
      if prevX is not None: gaplst += (x-prevX, )
      prevX = x
    avgap = sjmAvg(gaplst)
    rating = ()
    for i in range(0, len(Xlist)):
      x = Xlist[i]
      if len(regData[x]) > 3:
        ch,y,xc,dy = regData[x]
      else:
        ch,y,xc = regData[x]
        dy = 0
      g = 0
      if (i == 0):
        g = Xlist[1] - x
      if (i == len(Xlist) - 1):
        g = x - Xlist[-2]
      # if the leading / trailing char is I be more draconian...
      if (g > 0) and (ch == 'I'): g /= math.pow(xc, 2.5)
      if g > avgap:
        g -= avgap
        g *= 4 / avgap
      else: g = 0
      # calculate average XC but WITHOUT this current entry
      XClistNaN = XClist.copy()
      XClistNaN[i] = np.NaN
      axc = np.nanmean(XClistNaN)
      dxc = max(0, axc*axc - xc*xc) * 18
      q = 2*dy + dxc + g
      rating += (q, )
    wq = max(rating)
    if wq > thr:
      for i in range(0, len(Xlist)):
        if wq == rating[i]:
          del(regData[Xlist[i]])
          return True
    return False

  def gap2BestFit(xref, regData):
    Xlist = sorted(regData)
    # Pre-load the Xlist with duplicate entries - more duplicates for higher xcs
    del(Xlist[Xlist.index(xref)]) # but remove the specified entry
    x = Xlist.copy()
    for i in Xlist:
      xc = regData[i][2]
      nd = int(math.pow(2, max(0, (xc * xc * 12) - 3))) # 0.9 => +32 dups ... 0.65 => +2 dup
      if nd > 1:
        for n in range(1,nd): x.append(i)
    # Linear regression analysis on the augmented list
    y = [regData[i][1] for i in x]
    x_mean, y_mean = np.mean(x), np.mean(y)
    covar = 0.0
    for i in range(len(x)): covar += (x[i] - x_mean) * (y[i] - y_mean)
    b1 = covar / max(1, sum([(p - x_mean)**2 for p in x]))
    b0 = y_mean - b1 * x_mean
    dy = abs(regData[xref][1] - (b1 * xref + b0))
    return dy

  # Force a set of XY points to fit a straight line to within the specified coef-of-variance
  # Progressively delete poorest entries in regData until variance acheived
  # Augment each regData entry with it's dy error distance (in pixels) in index[3]
  # Return an estimated y centre-line and the max deviation from it
  def forceFit(regData, vmax):
    v = vmax+1
    while v > vmax:
      Xlist = sorted(regData)
      XYset = [[x, regData[x][1]] for x in Xlist]   # list of X,Y pairs for later
      # now reject the points that fit most poorly to the best straight line through the OTHER points
      worst = 0
      wi = 0
      i = 0
      rslt = []
      for x,y in XYset:
        dy = gap2BestFit(x, regData) # y gap to the best-fit line WITHOUT the current x point
        if dy > worst:
          worst = dy
          wi = len(rslt)
        rslt.append(dy)
        if len(regData[x]) < 4: regData[x].append(dy)
        else: regData[x][3] = dy
      v = np.var(rslt)
      if v > vmax:
        #print('FDel', Xlist[wi], regData[Xlist[wi]])
        del(regData[Xlist[wi]])
      # re-calc mean
      y_mean = 0
      for x in regData: y_mean += regData[x][1]
      y_mean /= len(regData)
    # want to return the max deviation from the int(y_mean)
    ycl = int(y_mean + 0.5)
    dy = ()
    for x in regData: dy += (abs(ycl - regData[x][1]), )
    return max(dy), ycl

  # Return index (x value) of nearest chars both left and right of target
  def nearestValids(x, regData):
    if x in regData: return (x,x)
    xs = sorted(regData) # smallest first
    if len(xs) == 0: return(None, None)
    if x < xs[0]: return (None, xs[0])
    if x > xs[-1]: return (xs[-1], None)
    for i in range(1, len(xs)):
      if (x < xs[i]) : break
    return (xs[i-1], xs[i])

  # Trim away blank (white) borders from an image
  # assumes at least one black pixel exists in image
  # and that the image has non-black borders on all 4 edges
  def trimImage(im):
    Hmins = np.amin(im, axis = 0) # list of min values in each V column
    # we can discard any left/right hand image colums that are nz in Hmins
    lhs = 0
    while Hmins[lhs] > 127: lhs += 1
    rhs = lhs
    while Hmins[rhs] < 128: rhs += 1
    Vmins = np.amin(im, axis = 1)
    # we can discard any top/bottom image rows that are nz in Vmins
    top = 0
    while Vmins[top] > 127: top += 1
    bot = top
    while Vmins[bot] < 128: bot += 1
    return im[top:bot, lhs:rhs]

  # Return a char template image to the specified char height & width to include extra
  # one or two pixel H & V borders so as to force an odd H & V pixel size as required by template matcher
  def makeCharTmplt(c, th, tw):
    im = cv2.resize(np.array(OCRFonts.UKFont[c], dtype=np.uint8), (tw, th), interpolation=cv2.INTER_CUBIC)
    # Add borders post-scale and force an odd-number of pixels in both H & V
    iw = tw + 6 - (1 - (tw % 2))
    ih = th + 6 - (1 - (th % 2))
    tmplt = np.ones((ih,iw), dtype=np.uint8)
    tmplt *= 255
    tmplt[2:th+2, 2:tw+2] = im
    return tmplt

  # Template-matches (XC) the img with character-templates from chlst made to the specified size (th,tw)
  # returns the average of the top n peaks found in the resulting xcmap
  # along with a dictionary of XC peak values indexed by character (from chlst)
  def preScan(img, AOI, chlst, th, tw, n, pw, chwI):
    xcmap = np.zeros(img.shape)
    chmap = np.zeros(img.shape, dtype=str)
    charMaxXC = {}
    x0,y0,x1,y1 = AOI
    subimg = img[y0:y1, x0:x1]
    if len(subimg) < 10: return 0,0,0,0,{},(),{}
    for c in chlst:
      tmplt = makeCharTmplt(c, th, tw) # tmplt will have guaranteed odd-numbered H & W pixel counts
      # make sure template is smaller (less high in y) than AOI
      while len(tmplt) >= len(subimg): tmplt = tmplt[1:-1]
      xo = x0 + len(tmplt[0]) // 2
      yo = y0 + len(tmplt) // 2
      xco = cv2.matchTemplate(subimg, tmplt, cv2.TM_CCOEFF_NORMED)
      charMaxXC[c] = np.amax(xco)
      #if DEBUG and (c == '3'):
      #  print(charMaxXC[c])
      #  cv2.imshow('xc_debug', xco)
      #  cv2.imshow('tmp_debug', tmplt)
      if charMaxXC[c] > 0.33:
        for x in range(0, len(xco[0])):
          for y in range(0, len(xco)):
            if xcmap[y+yo,x+xo] < xco[y,x]:
              xcmap[y+yo,x+xo] = xco[y,x]
              chmap[y+yo,x+xo] = c
    # find the peaks
    ppoints = findLocalPeaks(xcmap, tw * 1.1)
    if len(ppoints) == 0: return 0,0,0,0,{},(),{}
    # calc average xc of the top n peaks (n as specified)
    xcc = ()
    for x,y,xc in ppoints: xcc += (xc, )
    if len(xcc) > n:
      bxc = np.average(sorted(xcc, reverse = True)[0:n])
    else:
      bxc = np.average(xcc)
    # from ppoints make regdata
    regData = {}
    prevX = None
    for x,y,xc in ppoints:
      ch = chmap[y,x]
      if prevX is not None:
        # check for overlapping chars
        if x - prevX < tw + 1:
          # a conflict, pick only the strongest
          if xc < regData[prevX][2]: continue
          del(regData[prevX])
      regData[x] = [ch, y , xc]
      prevX = x
    # Reject any edge chars that are > max possible inter-char-gap pixels apart
    while delOOBChars(regData, pw, tw, chwI): pass
    # Reject the worst Y co-ordinate out-liers
    dy,yc = forceFit(regData, 0.66)
    return bxc, dy, yc, len(tmplt), charMaxXC, ppoints, regData

  # Run a simplified XC scan for I characters
  # we only scan along the yc centre-line
  def IScan(img, yc, tw, th):
    xcmap = np.zeros(img.shape)
    charMaxXC = {}
    tmplt = makeCharTmplt('I', th, tw) # tmplt will have guaranteed odd-numbered H & W pixel counts
    xo = len(tmplt[0]) // 2
    tmh = len(tmplt) // 2
    # make sure template is smaller (less high in y) than AOI
    im = img[(yc-1-tmh):(yc+2+tmh), 0:-1]
    if len(im) < 10: return ()
    while len(tmplt) >= len(im): tmplt = tmplt[1:-1]
    xco = cv2.matchTemplate(im, tmplt, cv2.TM_CCOEFF_NORMED)
    for x in range(0, len(xco[0])): xcmap[yc,x+xo] = max(xco[:,x])
    # find the peaks
    ppoints = findLocalPeaks(xcmap, tw * 1.1)
    if len(ppoints) == 0: return ()
    return ppoints

  # next integer in sequence 0,+1,+2,+3.. or -1,-2,-3..
  def nextUD(d):
    if d < 0: d -= 1
    else: d += 1
    return d

  # Rule-of-thumb conversion from a pre-scan bxc to a limit on character list length
  # roughly xcs < .8 do not limit character list thro to xc = 1: limit of top 8 chars
  def XC2N(xc):
    if xc < 0.8: return None
    return 8 + int((125 * (1 - xc)))

  # Derive a new (hopefully reduced) AOI from the latest set of ppoints XC peaks
  def AOIfromPks(iw, ih, tw, ppks, bxc, tph, dy, yc):
    xmin = None
    xmax = None
    for x,y,xc in ppks:
      if xc > 0.6 * bxc:
        if xmin is None: xmin = x # the first sig pk
        xmax = x # the last sig pk
    if xmin is None: xmin = 0
    if xmax is None: xmax = iw
    AOI[0] = max(0,    xmin - tw)
    AOI[2] = min(iw-1, xmax + tw)
    AOI[1] = max(0,    int(yc - 2*dy - (tph//2) + 2))
    AOI[3] = min(ih-1, int(yc + 2*dy + (tph//2) + 2))
    return AOI

  # Merge a post-optimisation rd with a master regData
  # Not for use with I chars
  def regMerge(regData, rd, tw):
    #rprint(regData)
    #rprint(rd)
    # Augment regData with rd data, resolving any conflicts by rejecting the weaker char
    for x in rd:
      if x in regData:
        if rd[x][2] > regData[x][2]: regData[x] = rd[x]
        continue
      (xxl,xxr) = nearestValids(x, regData)
      xxcl = xxcr = 0
      if xxl is not None:
        if x - xxl <= tw + 1: xxcl = regData[xxl][2]
      if xxr is not None:
        if xxr - x <= tw + 1: xxcr = regData[xxr][2]
      xxc = max(xxcl, xxcr)
      if rd[x][2] < xxc: continue
      if xxc > 0:
        if (xxcr > 0) and (xxcl > 0):
          del(regData[xxl])
          del(regData[xxr])
        else:
          if xxcl > xxcr: xx = xxl
          else: xx = xxr
          del(regData[xx])
      regData[x] = rd[x]
    return regData

  # Detect plate rotation angle from alignment of cas char detections
  def rotFromCas(XYWHs):
    if len(XYWHs) < 2: return 0,0
    x = []
    y = []
    # Linear regression analysis on the list
    for X,Y,W,H in XYWHs:
      x.append(X + W/2)
      y.append(Y + H/2)
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_span = max(x) - min(x)
    covar = 0.0
    for i in range(len(x)): covar += (x[i] - x_mean) * (y[i] - y_mean)
    b1 = covar / max(1, sum([(p - x_mean)**2 for p in x]))
    # b1 is slope, 0 = horiz, +ve = ?
    return math.atan(b1) * 180.0 / math.pi, x_span

  # Detect rotational angle of plate in image using hough lines
  def rotFromLines(img):
    # Find the longest (best) first few near-horizontal lines
    # scan down rapidly through length scale until we get > 3 results
    # then scan back up incrementally to find the first 3 H-lines
    PIBY180 = np.pi / 180
    ih,iw = img.shape
    llen = iw
    deglst = ()
    while (len(deglst) < 3) and (llen > 10):
      llen = int(llen * 0.9)
      lines = cv2.HoughLines(img, 1, PIBY180, llen)
      if lines is None: continue
      else:
        deglst = ()
        for line in lines:
          d, theta = line[0]
          degs = int(0.5 + (theta / PIBY180)) - 90
          dc = d + (iw * math.sin(degs * PIBY180) / 2)
          if (degs > -10) and (degs < 10) and (dc > ih * 0.1) and (dc < ih * 0.9):   # look only at mid, near horizontals
            deglst += (degs,)
    keep = deglst
    while len(deglst) > 3:
      llen += 1
      lines = cv2.HoughLines(img, 1, PIBY180, llen)
      if lines is None:
        deglst = keep      
        break
      else:
        keep = deglst
        deglst = ()
        for line in lines:
          d, theta = line[0]
          degs = int(0.5 + (theta / PIBY180)) - 90
          dc = d + (iw * math.sin(degs * PIBY180) / 2)
          if (degs > -10) and (degs < 10) and (dc > ih * 0.1) and (dc < ih * 0.9):   # look only at mid, near horizontals
            deglst += (degs,)  
    if len(deglst) == 0: return 0,0
    degs = votedAverage(deglst)   # -ve result means anticlockwise rotation
    v = np.var(deglst)
    cnf = max(0, (1 - v)*0.66)  # never that good!
    return degs, cnf

  # Detect the text skew angle using hough lines 
  def skewAngle(img, th):
    PIBY180 = np.pi / 180
    ih,iw = img.shape
    llen = th
    deglst = ()
    while len(deglst) < 32:
      deglst = ()
      llen = int(llen * 0.95) - 1
      lines = cv2.HoughLines(img, 1, PIBY180, llen)
      if lines is None: continue
      else:
        for line in lines:
          d, theta = line[0]
          degs = int(0.5 + theta / PIBY180)
          if degs > 90: degs -= 180         # convert to a continuum
          if abs(degs) < 9:    # pick only the near-verticals
            dc = abs(d) + (ih/2) * math.sin(degs * PIBY180)
            if (dc < iw * 0.1) or (dc > iw * 0.9): continue  # reject anything at edges
            deglst += (degs, )
    if len(deglst) < 6: return 0, 0
    return np.median(deglst), 1  # works better than: sjmAvg2(deglst, 2), 1

  # Return the likelyhood value (from PlateLUT) of the supplied reg
  # pick best 0/1, O/I interpretation
  # Return -ve if there's no match to any format template
  def likelyhoodScore(reg):
    creg = characterise(reg, 1)
    keys = PlateLUT.keys()
    score = -20
    for fmt in keys:
      if dotMatch(creg, fmt): score = max(score, PlateLUT[fmt])
    return score

  # Assuming a -ve likelyhood score for the given reg, can we find a single mis-read
  # character that is causing a mis-match to a valid format template?
  # If so then, further, if that char is commonly mis-read we simply swap it for it's doppleganger.
  # Otherwise return ''
  def swapBadCh(reg):
    creg = characterise(reg, 1)
    keys = list(PlateLUT.keys())
    for k in range(len(keys)-1, -1, -1):
      fmt = keys[k]
      if len(fmt) == len(creg):
        badCnt = 0
        for i in range(0, len(fmt)):
          if creg[i] == '.': continue
          if creg[i] != fmt[i]:
            badCnt += 1
            badi = i
        if badCnt == 0: return reg
        if badCnt == 1:
          # dict of commonly mis-read alpha/num pairs
          # note that most are reversible but O/0 equivalence means D & Q have a one-way association with 0 
          SwapTab = {'8':'B', 'B':'8', 'Z':'7', '7':'Z', '6':'G', 'G':'6', '5':'S', 'S':'5', 'D':'0', 'Q':'0'}
          c = reg[badi]
          if c in SwapTab: return reg[:badi] + SwapTab[c] + reg[badi+1:]
    return ''

  # convert reg into @#.? format where # maps to any number, @ maps to any u/c alpha,
  # dot maps to 0,1,O or I and ? mops up anything invalid that shouldn't be there.
  # If argument dots = 0 then 01IO->dot mapping is not done
  def characterise(reg, dots=1):
    creg = ''
    for i in range(0, len(reg)):
      ch = reg[i]
      if (ch in 'IO01') and (dots != 0): ch = '.'
      else:
        if ch in '0123456789': ch = '#'
        else:
          if ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ': ch = '@'
          else: ch = '?'
      creg += ch
    return creg

  # see if two strings are the same BUT with dot as a wild-card char
  def dotMatch(str1, str2):
    if str1 == str2: return True
    if len(str1) != len(str2): return False
    for i in range(0, len(str1)):
      if (str1[i] != '.') and (str2[i] != '.') and (str1[i] != str2[i]): return False
    return True

  # Massage 0/O and I/1 characters in supplied reg to match most likely UK format
  def massage(reg):
    creg = characterise(reg, 1)
    if '?' in creg: return reg
    if '.' not in creg: return reg
    keys = PlateLUT.keys()
    hiscore = -1
    hifmt = ''
    for fmt in keys:
      if dotMatch(creg, fmt):
        if PlateLUT[fmt] > hiscore:
          hiscore = PlateLUT[fmt]
          hifmt = fmt
    if hifmt == '': return reg
    # do the massage
    ro = ''
    for i in range(0, len(creg)):
      ch = reg[i]
      if creg[i] == '.':
        if hifmt[i] == '@':
          if ch == '0': ch = 'O'
          elif ch == '1': ch = 'I'
        if hifmt[i] == '#':
          if ch == 'O': ch = '0'
          elif ch == 'I': ch = '1'
      ro += ch
    return ro

  # Return plate bounding-box XYWH from regData and initial crop XY info
  # Note that the bbox ignores any rotation / de-skew transformations
  def bbFromRD(rd,h,w,crop):
    X,Y = crop
    Xlist = sorted(rd)
    X = max(0, X + Xlist[0] - w)
    Y = max(0, Y + rd[Xlist[0]][1] - 3*h//4)
    W = Xlist[-1] - Xlist[0] + 2*w
    H = 3*h//2
    return (X,Y,W,H)

  # scan through a dictionary, look for longest index string length
  def maxlen(dctn):
    mx = 0
    for k in dctn: mx = max(mx, len(k))
    return mx

  # ------------------ main --------------------

  t0 = time.time()
  if DEBUG: print(">>> start @ {:1.1f}".format(time.time() - t0))

  PIBY180 = np.pi / 180
  CLAHE = cv2.createCLAHE(clipLimit=1, tileGridSize = (8,8))     # invoke a contarst-limited adaptive histogram equaliser

  # Convert input image to gry-scale if necessary, look for a license plate, crop source image.
  if len(in_img.shape) > 2:
    img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    pcol = ''  # detect plate colour later
  else:
    img = in_img
    pcol = '-'  # plate colour: - => b/w image - can't detect b/g colour
  if (not isinstance(plateXYWH, tuple)) or (len(plateXYWH) != 4):
    rect = lookForPlate(img, TargetWH) # assumes a gryscale image
  else:
    rect = plateXYWH
  px,py,pw,ph = rect
  if pw == 0: return '!No Plate' 
  rect1 = enlargeCrop(rect) # expand the AOI
  th = int(0.5 + ph/PH2TH)
  tw = int(0.5 + ASPECT*ph/PH2TH)
  imgry = cropIt(img, rect1)
  cropXY = (rect1[0], rect1[1])
  ih,iw = imgry.shape
  if DEBUG: print('Source img cropped to:', ih, 'x', iw)
  # Histogram-eq & de-noise
  img_gry = CLAHE.apply(imgry)
  img_gry = cv2.fastNlMeansDenoising(img_gry, None, 16, 7, 21)
  img_gry = CLAHE.apply(img_gry)
  # Neg detect, some plates are white-on-black, negate the image if approp
  x0,y0,x1,y1 = (iw-pw)//2, (ih-ph)//2,(iw+pw)//2, (ih+ph)//2, # define a smaller AOI
  img_t = cv2.adaptiveThreshold(img_gry[y0:y1,x0:x1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, th + (1 - (th % 2)), 2)
  if np.mean(img_t) < 128: # neg detection
    img_gry = 255 - img_gry
    imgry   = 255 - imgry
    pcol = 'B'
  # Adaptive threshold & edge detect ready for hough lines
  img_thr = cv2.adaptiveThreshold(img_gry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, th + (1 - (th % 2)), 2)
  skel = cv2.Canny(img_thr, 100, 200)
  if DEBUG: skel0 = skel.copy() # keep a copy of original

  if DEBUG: print(">>> start horizontalisation @ {:1.1f}".format(time.time() - t0))

  # Horizontalisation...
  # Detect rotation...
  # 1. using SJM's Haar-Cascade character spotter
  chsList, Yav = seekChrs(img_gry, (tw,th))
  rfc, xspan = rotFromCas(chsList)
  ccnf = min(1, xspan/pw) # confidence range 0 .. 1
  # 2. using Hough lines
  rfl, lcnf = rotFromLines(skel)
  # combine results
  if lcnf + ccnf > 0:
    # average results according to relative confidences   
    degs = ((rfc * ccnf) + (rfl * lcnf)) / (ccnf + lcnf) 
    if DEBUG: print('Detected rotation: {:0.2f}'.format(degs))
    if abs(degs) >= 0.15:
      if DEBUG: print('Fixing rotation...')
      rmx = cv2.getRotationMatrix2D((iw//2,ih//2),degs,1)
      img_gry = cv2.warpAffine(img_gry, rmx, (iw,ih))
      imgry   = cv2.warpAffine(imgry, rmx, (iw,ih))
      # re-make skel fro de-skew
      img_thr = cv2.adaptiveThreshold(img_gry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, th + (1 - (th % 2)), 2)
      skel = cv2.Canny(img_thr, 100, 200)
  else:
    if DEBUG: print("Can't determine rotation")

  if DEBUG: print(">>> start de-skew @ {:1.1f}".format(time.time() - t0))

  # De-skew...
  # Find skew angle...
  degs, cnf = skewAngle(skel, th)
  if cnf > 0:
    if DEBUG: print('Detected skew: {:0.2f}'.format(degs))
    # -ve result means chars slope backwards
    d = ih * math.tan(degs * PIBY180) # deskew amount in pixels
    if abs(d) >= .5:
      # apply skew correction
      if DEBUG: print('Fixing skew...')
      xl,xr = int(iw * 0.1), int(ih * 0.9)
      pts1 = np.float32([[xl,  0],[xr,  0],[0,ih-1],[iw-1,ih-1]]) # 4 source points
      pts2 = np.float32([[xl-d,0],[xr-d,0],[0,ih-1],[iw-1,ih-1]]) # 4 destn points
      PT = cv2.getPerspectiveTransform(pts1,pts2)
      img_gry = cv2.warpPerspective(img_gry, PT, (iw, ih))
      imgry   = cv2.warpPerspective(imgry, PT, (iw, ih))

  if DEBUG: print(">>> start enhance @ {:1.1f}".format(time.time() - t0))

  # More image enhancements...
  img_thr = cv2.adaptiveThreshold(img_gry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, th + (1 - (th % 2)), 2)
  SE = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))  
  img_t2 = cv2.dilate(img_thr, SE)
  img_proc = cv2.bitwise_and(img_gry, img_t2)
  img_t3 = cv2.erode(img_thr, SE)
  img_proc = cv2.bitwise_or(img_proc, img_t3)
  img_proc = cv2.blur(img_proc, (3,3))

  if DEBUG: print(">>> start char height estimation @ {:1.1f}".format(time.time() - t0))

  # Improve estimate of character height...
  # Also improve estimate of plate position and size
  # Method 1. Identify character bboxes using UKChar haar-cascade classifyer
  chsList, Yav = seekChrs(img_gry, (tw,th))
  if len(chsList) < 5:  # maybe some more processing would help
    if DEBUG: print('Found only ',len(chsList),'chars, re-try with thresholding...')
    chsListB, YavB = seekChrs(img_proc, (tw,th))
    if len(chsListB) > len(chsList):
      chsList = chsListB
      Yav = YavB
  if DEBUG: print('Found', len(chsList), 'UKChars')
  # Compute centre-line, char height (and horizontal range for later)
  ycl = ih // 2
  Hlst = (th, )
  xl, xr = int(iw * 0.45), int(iw * 0.55)
  if (len(chsList) > 0):
    ycl = int(Yav + 0.5)
    for n in range(0, len(chsList)):
      Hlst += (chsList[n][3] * CAS_OVER, ) # HEIGHT compensated for over-estimation
      xl = min(xl, chsList[n][0])
      xr = max(xr, chsList[n][0]+chsList[n][2])
  if len(chsList) < 2: xl, xr = int(iw * 0.25), int(iw * 0.75)
  # Method 2. Locate & measure text-like blocks using the sobel filter
  sox = cv2.Sobel(img_proc, cv2.CV_8U, 1, 0, ksize=1)
  _,thx = cv2.threshold(sox, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  SE = cv2.getStructuringElement(cv2.MORPH_RECT, (int(pw/3.5), int(ph/11)))
  morf=cv2.morphologyEx(thx, cv2.MORPH_CLOSE, SE)
  # measure text height at several mid-ish points along x-axis
  ths,ycls = measureTxH(morf, xl, xr)
  # Average both methods' data
  n = len(chsList)
  if (n > 0) and (ycls > 0): ycl = int(0.5 + (((ycl * n) + ycls) / (n + 1)))
  elif ycls > 0: ycl = ycls
  if (ths > 0): Hlst += (ths, )
  if len(Hlst) > 0: th = int(np.average(Hlst) + 0.5)
  if th > ih * 0.9: return '!Bad plate - text too big'
  if th < 20: return '!Bad plate - text too small'
  if DEBUG: print("Char height estimate:", th, "px")
  if DEBUG: print(">>> start contrast stretch @ {:1.1f}".format(time.time() - t0))

  # Contrast-stretch(5%) the whole image using a histogram analysis from
  # just the central-th x central-coverage-width known plate area... 
  # First, create the AOI mask image
  mask = np.zeros((ih,iw), dtype=np.uint8)
  x1,y1,x2,y2 = xl, int(ycl-(th/2)+1), xr, int(ycl+(th/2)-1)
  mask[y1:y2,x1:x2] = 255
  # calc the histogram...
  hist = cv2.calcHist([cv2.blur(imgry, (3,3))], [0], mask, [256], [0,256])  # get histogram of masked area
  newLUT = heq(hist)  # equalise it
  for x in range(0, iw):  # apply
    for y in range(0, ih): imgry[y][x] = newLUT[imgry[y][x]]

  # Prepare an image source for the XC stages...
  # Blurring helps to reduce the unwelcome image sharpening effects from the camera 
  img_xc = cv2.blur(imgry, (3,3))

  # re-evaluate plate extents for initial XC AOI
  pw = int(PW2CH*th + 0.5)
  tw = int(ASPECT*th + 0.5)
  xl, xr = tw//2, iw-1-tw//2
  # Define initial AOI to height th*2 around ycl and width to xl..xr horiz extents
  AOI = [max(0, xl-(tw//2)), max(0, ycl-th), min(iw, xr+(tw//2)), min(ih, ycl+th)] # X1,Y1,X2,Y2

  if DEBUG: print(">>> start XC stages @ {:1.1f}".format(time.time() - t0))

  # XC OCR stages with character-size optimisation...
  # We template-match to a range of text sizes and shapes to try to find the optimum by
  # itteratively adjusting template th and tw for maximum peak-heights in the XC landscape
  # This is the real time-consuming stage so we do some tricks to minimise delays.
  # Note we exclude the I character here due to its anomalous width, see later
  n = max(4, len(chsList))  # the number of chars to pick from the top of the list when evaluating overall match quality
  chs = '23456789ABCDEFGHJKLMNOPQRSTUVWXYZ' # Valid UK plate characters w/o I
  base_xc, cv, yc, tmph, cxcLst, ppoints, regData = preScan(img_xc, AOI, chs, th, tw, n, pw, 0) # base measure
  ycl = int(yc + 0.5)
  # Increase speed by scanning a reduced character set and AOI
  # The poorer the over-all XC the more chars we scan for & the wider the AOI (& the slower the process)
  chs = sorted(cxcLst, key=cxcLst.__getitem__, reverse=True)[0:XC2N(base_xc)]
  AOI = AOIfromPks(iw, ih, tw, ppoints, base_xc, tmph, cv, ycl)
  if DEBUG: print('base xc:', base_xc)
  if DEBUG: print(chs)
  d = 1
  tho = th

  if DEBUG: print(">>> start XC height opt @ {:1.1f}".format(time.time() - t0))

  # text height optimisation...
  while True:
    if DEBUG: print("H-opt loop")
    xc, cv, yc, tmph, cxcLst, ppoints, rd = preScan(img_xc, AOI, chs, th+d, int(((th+d) * ASPECT) + 0.5), n, pw, 0)
    regData = regMerge(regData, rd, tw)
    if xc <= base_xc:
      if d != 1: break
      d = -1
      continue
    if DEBUG: print('Improved th:', d, '=>', xc)
    chs = sorted(cxcLst, key=cxcLst.__getitem__, reverse=True)[0:XC2N(base_xc)]
    if DEBUG: print(chs)
    base_xc = xc
    ycl = int(yc + 0.5)
    AOI = AOIfromPks(iw, ih, tw, ppoints, base_xc, tmph+d, cv, ycl)
    tho = th+d
    pw = int(PW2CH * tho + 0.5)
    d = nextUD(d)
  th = tho

  if DEBUG: print(">>> start XC width opt @ {:1.1f}".format(time.time() - t0))

  # now width...
  n = min(7, min(3, len(regData)))
  tw = int((th * ASPECT) + 0.5) # starting point, based on best th
  d = 1
  two = tw
  while True:
    if DEBUG: print("W-opt loop")
    xc, cv, yc, tmph, cxcLst, ppoints, rd = preScan(img_xc, AOI, chs, th, tw+d, n, pw, 0)
    regData = regMerge(regData, rd, tw)
    if xc <= base_xc:
      if d != 1: break
      d = -1
      continue
    chs = sorted(cxcLst, key=cxcLst.__getitem__, reverse=True)[0:XC2N(base_xc)]
    base_xc = xc
    ycl = int(yc + 0.5)
    AOI = AOIfromPks(iw, ih, tw, ppoints, base_xc, tmph, cv, ycl)
    two = tw+d
    pw = int(STW2TW * two)  # better approximation to license-string width
    d = nextUD(d)
  tw = two

  if len(regData) == 0: return '!No reg data'

  if DEBUG: print(">>> start I stage prep @ {:1.1f}".format(time.time() - t0))

  pw = int(STW2TW * tw)  # better approximation to total character-string width
  twI = int(tw * ASPI + 0.5)

  # Reject weak chars on the basis of an over-all quality score
  # Assumes forceFit() has previously run to extend the regData dict
  while rejectWeaklings(regData, 6.66): pass

  if DEBUG: pprint.pprint(regData)

  if len(regData) == 0: return '!No reg data'

  # re-calc base-xc & ycl on the remaining chars in prep for I stage
  base_xc = 0
  ycl = 0
  for x in regData:
    ycl += regData[x][1]
    base_xc += regData[x][2]
  base_xc /= len(regData)
  ycl = int(0.5 + ycl/len(regData))

  # Use a simpler centre-line scan for spotting I chars...
  ipoints = IScan(img_xc, ycl, twI, th)
  # Augment regData with the new I-spots resolving any conflicts by deleting the weaker char
  n = 0
  for x,y,xc in ipoints:
    if xc < base_xc * 0.6: continue     # I xc must be in-line with other xcs
    if x in regData:
      if xc > regData[x][2]: regData[x] = ['I', y, xc]
      continue
    (xxl,xxr) = nearestValids(x, regData)
    xxcl = xxcr = 0
    if xxl is not None:
      if regData[xxl][0] == 'I': cw = twI
      else: cw = tw
      if (x - xxl) <= (2 + cw + twI)/2: xxcl = regData[xxl][2]
    if xxr is not None:
      if (xxr - x) <= (2 + tw + twI)/2: xxcr = regData[xxr][2]
    xxc = max(xxcl, xxcr)
    # Conflict: strongest XC wins but we bias the contest against I chars
    if math.pow(xc,1.5) < xxc: continue
    if xxc > 0:
      # new char is stronger, need to delete something.
      if (xxcr > 0) and (xxcl > 0):
        del(regData[xxl])  # delete the incumbants
        del(regData[xxr])
      else:
        if xxcl > xxcr : xx = xxl
        else: xx = xxr
        del(regData[xx])  # delete the incumbant
    regData[x] = ['I', y, xc]
    n += 1
  if n > 0:
    if DEBUG: pprint.pprint(regData)
    # re-run rejection processes on I-extended regData
    while delOOBChars(regData, pw, tw, twI): pass
    forceFit(regData, 1.33)
    while rejectWeaklings(regData, 8): pass

  if DEBUG: print(">>> start finish-up @ {:1.1f}".format(time.time() - t0))

  # if necessary reduce to the max allowed char count...
  w = 8.0
  mxln = maxlen(PlateLUT)
  while len(regData) > mxln:
    while not rejectWeaklings(regData, w): w *= 0.95

  regn = ''
  for rd in sorted(regData): regn += regData[rd][0]
  if regn.replace('I', '') == '': return '!No reg data'
  # Match to most likely UK reg format
  scr = likelyhoodScore(regn) # -20, 0 ... +20
  # compute a composite confidence factor 0..99, part base_xc, part likelyhoodness
  cnf = 0
  if scr < 0:
    # NOT a valid format!
    # If the error can be narrowed down to a single char then
    #  a commonly mis-read alpha/num assoc could be swapped out:
    #  ie: 8 <=> B, 5 <=> S, D => 0, Q => 0, 7 <=> Z, 6 <=> G
    #  if so return a reduced conf < 80
    # Else Return a cnf of zero to indicate a non-confoming format
    # &&& Perhaps also challenge leading / trailing I chars
    alt = swapBadCh(regn)
    if alt != '':
      regn = alt
      cnf = int((base_xc * 0.8) * 100) # reduced conf
  else: cnf = int((base_xc * 0.9 + scr * 0.004) * 100 + 0.5) # composite conf
  regn = massage(regn)  # optimise 0/O, I/1 choices 

  # calc plate bbox relative to source image
  bbox = bbFromRD(regData, th, tw, cropXY) # XYWH

  # get plate b/g colour if possible...
  if len(pcol) == 0:
    # we have a colour source image, and it's not a wht-on-blk plate
    pcol = WhtOrYel(in_img, bbox)  # returns Y or W or '-' if unsure
  # else pcol is already either '-' for unknowable or 'B' for black

  if DEBUG: print(">>> end @ {:1.1f}".format(time.time() - t0))

  return regn, cnf, bbox, pcol

# ----------------- demo mode? Accept an image file name -------------------

if __name__ == "__main__":
  import sys
  if len(sys.argv) > 1:
    img = cv2.imread(sys.argv[1])
    DEBUG = True
    print(ReadPlate(img, (170,50)))
  else: print('Please provide an image file name, UKRegOCR will try to find and read a UK number-plate')  

