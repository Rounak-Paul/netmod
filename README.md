# netmod
## Rounak's Python Module for Automating basic fucntions

## Documentation:

### Module: Image Processing </br>

  img = cv2.imread(path) </br>
  video = cv2.VideoCapture(path) </br>
  rgb = np.array([R,G,B]) </br>
  
  <ul>
  <li><b>video2img(video,dir):</b> Video saved as img frame by frame</li> </br>
  
  <li><b>gray(img):</b> Grayscale image from BGR</li> </br>
  
  <li><b>colorFilterUI(img):</b> Filter RGB image with a pop-up color picker UI</li> </br>
  
  <li><b>colorFilter(img,rgb):</b> Takes numpy array like defined above with each enty between and including 0 to 255</li> </br>
  
  <li><b>sharp(img):</b> Sharpens image</li> </br>
  
  <li><b>smooth(img):</b> Gaussian Blur</li> </br>
  
  <li><b>layerExtract(img,int):</b> Extracts each 3 layers of RGB image, value of <b>int</b> is in range 0, 1, 2 for R, G, B respectively</li> </br>
  
  <li><b>edge(img):</b> Extracts edge, Laplacian</li> </br>
  
  <li><b>powerlaw(img,int):</b> Powerlaw Transformation, value of <b>int</b> is the gamma</li> </br> 
  
  </ul>
  




