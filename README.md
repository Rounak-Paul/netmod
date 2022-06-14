# netmod
## Rounak's Python Module for Automating basic fucntions

## Documentation:

### Module: Image Processing </br>

  img = cv2.imread(path) </br>
  video = cv2.VideoCapture(path) </br>
  rgb = np.array([R,G,B]) </br>
  
  <b>video2img(video,dir):</b> Video saved as img frame by frame </br>
  
  <b>gray(img):</b> Grayscale image from BGR </br>
  
  <b>colorFilterUI(img):</b> Filter RGB image with a pop-up color picker UI </br>
  
  <b>colorFilter(img,rgb):</b> Takes numpy array like defined above with each enty between and including 0 to 255 </br>
  
  <b>sharp(img):</b> Sharpens image</br>
  
  <b>smooth(img):</b> Gaussian Blur</br>
  
  <b>layerExtract(img,int):</b> Extracts each 3 layers of RGB image, value of <b>int</b> is in range 0, 1, 2 for R, G, B respectively</br>
  
  <b>edge(img):</b> Extracts edge, Laplacian</br>
  
  <b>powerlaw(img,int):</b> Powerlaw Transformation, value of <b>int</b> is the gamma
  
  
  
  
  




