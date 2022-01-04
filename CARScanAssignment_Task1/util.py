import pandas as pd
import requests
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

#Function for output of desired visualization
def carVisulaization(image_path, meta_data_json_path):
    #imoprting the meta json data from github along with reading the image from image path.
    data=requests.get(meta_data_json_path)
    img=cv2.imread(image_path)
    resultImageList=[]
    
    #converting json data into a dataframe for small preprocessing.
    jsdata=data.json()
    df=pd.DataFrame.from_dict(jsdata)
    
    l=len(df)
    m=0
    flag=False
    if df['type'][0]=='choices':
        m=1
        flag=True
    
    #Denormalizing the points.
    for i in range(m,l):
        for key in df['value'][i]:
            if key=='points':
                for j in range(len(df['value'][i][key])):
                    df['value'][i][key][j]=denormalize(df['value'][i][key][j],df['original_width'][i],df['original_height'][i])
    
    #Taking the points corresponding to their label into 2 different arrays for futher process.
    pts=[[] for i in range(m,l)]
    label=[]
    for i in range(m,l):
        for key in df['value'][i]:
            if key=='points':
                for j in range(len(df['value'][i][key])):
                    pts[i-1].append((math.floor(df['value'][i][key][j][0]),math.floor(df['value'][i][key][j][1])))
            else:
                label.append(df['value'][i][key][0])
    
    #Filling the polygon using points and making them transparent by 25% by changing their alpha values.
    for i in range(m,l): 
        overlay=img.copy()
        out=img.copy()
        if flag:
            lab=label[i-1]
            arr=np.array(pts[i-1])
        else:
            lab=label[i]
            arr=np.array(pts[i])
        if lab=='BackaboveFender':
            cv2.fillPoly(overlay,[arr],color=(0,0,255))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(0,0,255),thickness=1)
        elif lab=='Bonnet':
            cv2.fillPoly(overlay,[arr],color=(255,0,0))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(255,0,0),thickness=1)
        elif lab=='Bumper':
            cv2.fillPoly(overlay,[arr],color=(0,255,0))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(0,255,0),thickness=1)
        elif lab=='Door':
            cv2.fillPoly(overlay,[arr],color=(255,255,0))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(255,255,0),thickness=1)
        elif lab=='Fender':
            cv2.fillPoly(overlay,[arr],color=(255,0,255))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(255,0,255),thickness=1)
        elif lab=='Light':
            cv2.fillPoly(overlay,[arr],color=(255,200,200))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(255,200,200),thickness=1)
        elif lab=='Mirror':
            cv2.fillPoly(overlay,[arr],color=(200,255,200))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(200,255,200),thickness=1)
        elif lab=='RockerPanel':
            cv2.fillPoly(overlay,[arr],color=(150,150,150))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(150,150,150),thickness=1)
        elif lab=='Roof':
            cv2.fillPoly(overlay,[arr],color=(150,200,150))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(150,200,150),thickness=1)
        elif lab=='Wheel':
            cv2.fillPoly(overlay,[arr],color=(100,150,100))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(100,150,100),thickness=1)
        elif lab=='WindowPanel':
            cv2.fillPoly(overlay,[arr],color=(150,150,200))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(150,150,200),thickness=1)
        elif lab=='Boot':
            cv2.fillPoly(overlay,[arr],color=(50,10,20))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(50,10,20),thickness=1)
        elif lab=='Dent&Scratch':
            cv2.fillPoly(overlay,[arr],color=(0,0,255))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(0,0,255),thickness=1)
        elif lab=='Dent&Scratch(zoom)':
            cv2.fillPoly(overlay,[arr],color=(255,0,255))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(255,0,255),thickness=1)
        elif lab=='Broken':
            cv2.polylines(out,[arr],isClosed=True,color=(150,200,150),thickness=1)
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(150,200,150),thickness=1)
        else:
            cv2.fillPoly(overlay,[arr],color=(200,150,200))
            cv2.addWeighted(overlay,0.25,out,0.75,0,out)
            cv2.polylines(out,[arr],isClosed=True,color=(200,150,200),thickness=1)
        img=out.copy()
        
    resultImageList.append(img)
    
    #Creating a rectangle over the polygon along with their label
    for i in range(m,l): 
        out=img.copy()
        if flag:
            lab=label[i-1]
            arr=np.array(pts[i-1])
        else:
            lab=label[i]
            arr=np.array(pts[i])
        out1=makeRectwithLabel(out,arr,lab)
        img=out1.copy()
    
    resultImageList.append(img)
    #returning the final images
    return resultImageList


"""--------------------------------------------------------------------------------------------------------------------------"""



#Function for denormalizing the coordinates, normalized_x=(x/image_width)*100, normalized_y=(y/image_height)*100
def denormalize(points,original_width,original_height):
        new_x=points[0]*original_width/100
        new_y=points[1]*original_height/100
        return [new_x,new_y]
    

    
"""--------------------------------------------------------------------------------------------------------------------------"""



#Function for creating the rectangle over the polygon with the label highlighted
def makeRectwithLabel(img,pts,lab):
    a=np.array(pts)
    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(mask, a, 1)
    mask = mask.astype(np.bool)
    out1 = np.zeros_like(img)
    out1[mask] = img[mask]
    
    #Main crux of the visualization
    x_max,x_min,y_max,y_min,curr_x,curr_y=0,1000,0,1000,0,0
    for i in range(len(pts)):
        curr_x=pts[i][0]
        curr_y=pts[i][1]
        x_max=max(x_max,curr_x)
        x_min=min(x_min,curr_x)
        y_max=max(y_max,curr_y)
        y_min=min(y_min,curr_y)

    (width,height),_=cv2.getTextSize(lab,cv2.FONT_ITALIC,0.4,1)
    if lab=='BackaboveFender':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (0,0,255), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (0,0,255), 1)
    elif lab=='Bonnet':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (255,0,0), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (255,0,0), 1)
    elif lab=='Bumper':
        out1 = cv2.rectangle(out1,(x_min, y_min), (x_max, y_max), (0,255,0), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (0,255,0), 1)
    elif lab=='Door':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (255,255,0), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (255,255,0), 1)
    elif lab=='Fender':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (255,0,255), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (255,0,255), 1)
    elif lab=='Light':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (255,200,200), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (255,200,200), 1)
    elif lab=='Mirror':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (200,255,200), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (200,255,200), 1)
    elif lab=='RockerPanel':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (150,150,150), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (150,150,150), 1)
    elif lab=='Roof':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (150,200,150), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (150,200,150), 1)
    elif lab=='Wheel':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (100,150,100), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (0,100,0), 1)
    elif lab=='WindowPanel':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (100,100,200), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (100,100,200), 1)
    elif lab=='Boot':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (50,10,20), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (50,10,20), 1)
    elif lab=='Dent&Scratch':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (0,0,255), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (0,0,255), 1)
    elif lab=='Dent&Scratch(zoom)':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (255,0,255), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (255,0,255), 1)
    elif lab=='Broken':
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (150,200,150), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (150,200,150), 1)
    else:
        out1 = cv2.rectangle(out1, (x_min, y_min), (x_max, y_max), (200,150,200), 1)
        points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
        points=np.array(points)
        out1=cv2.fillPoly(out1,points,color=(255,255,255))
        cv2.putText(out1, lab, (x_min, y_min+10), cv2.FONT_ITALIC, 0.4, (200,150,200), 1)
    final=cv2.bitwise_or(img,out1)
    return final

