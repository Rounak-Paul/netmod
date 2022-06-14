# netmod
## Rounak's Python Module for Automating basic fucntions

## Documentation:

### Module: Image Processing </br>

  img = cv2.imread(path) </br>
  video = cv2.VideoCapture(path) </br>
  rgb = np.array([R,G,B]) </br>
  
  <li>
  <ul><b>video2img(video,dir):</b> Video saved as img frame by frame </br></ul>
  
  <ul><b>gray(img):</b> Grayscale image from BGR </br></ul>
  
  <ul><b>colorFilterUI(img):</b> Filter RGB image with a pop-up color picker UI </br></ul>
  
  <ul><b>colorFilter(img,rgb):</b> Takes numpy array like defined above with each enty between and including 0 to 255 </br></ul>
  
  <ul><b>sharp(img):</b> Sharpens image </br></ul>
  
  <ul><b>smooth(img):</b> Gaussian Blur </br></ul>
  
  <ul><b>layerExtract(img,int):</b> Extracts each 3 layers of RGB image, value of <b>int</b> is in range 0, 1, 2 for R, G, B respectively </br></ul>
  
  <ul><b>edge(img):</b> Extracts edge, Laplacian </br></ul>
  
  <ul><b>powerlaw(img,int):</b> Powerlaw Transformation, value of <b>int</b> is the gamma </br>   </ul>
  </li>
  
  
  




