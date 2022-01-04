from util import *

#Please paste the image path.
imgPath=r" "
#For example --> imgPath=r"C:/Users/Home/OneDrive/Desktop/5.jpg"

#Please paste the json file link from the github.
meta_data_json_path=" "
#For example --> meta_data_json_path="https://raw.githubusercontent.com/PushpakBhoge512/Assignment/main/Data%20Visualization/data/5.json"

imageList=carVisulaization(image_path=imgPath,meta_data_json_path=meta_data_json_path)

cv2.imshow("Sub-Task-1.1",imageList[0])
cv2.imshow("Sub-Task-1.2",imageList[1])
cv2.waitKey(0)





