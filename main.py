import numpy as np
import cv2
import torch
from PIL import Image
import open_clip
import keyboard

#load OpenAI CLIP model and tokenizer architecture, and load pretrained model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

brand_list = ["nike logo", "adidas logo", "under armour logo", "not a logo"]
brand_queue = []

def from_ndarray_to_PIL(ndarray_img):
  #Converts ndArray image into PIL
  #image webcam ndArray
  return Image.fromarray(ndarray_img)

def classify_brand(pil_img):
  #Takes PIL image and gets it's brand name. 
  image = preprocess(pil_img).unsqueeze(0)
  text = tokenizer(brand_list)
  with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

  return brand_list[np.argmax(text_probs.cpu().detach().numpy())]

cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read() # numpy array 
    brand_name = classify_brand(from_ndarray_to_PIL(img))
    
    #append to brand_name array so they do not append same brand in a row
    if brand_name != "not a logo" :
      if len(brand_queue) != 0:
        if brand_name != brand_queue[-1]:
          brand_queue.append(brand_name)
      else:
        brand_queue.append(brand_name)
    
    print(f"Output: {brand_name}")
    
    """
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    """
    if keyboard.is_pressed('esc'):
      break

cap.release()
cv2.destroyAllWindows()

#enumerate keep track of the number of iterations in a loop.
#when program quit, it prints the brand_queue
for i, name in enumerate(brand_queue):
  print(f"{i+1} :: {name}")

  """
recognition part is done and shared in Onedrive. 
you have to run this in python 3.10 (3.11 is the latest). 
To run, download libraries first, on CMD, 'pip install open_clip_torch' 'pip install torch' 'pip install opencv-python' 'pip install numpy' 
On VS code, open terminal change Power Shell to CMD, change the directory and 'python main.py' to run the code
when first time running the code on a device, it will download some pre-trained variables (only first time on a device)
As it is running it will save the brand name on 'brand_queue'.
press 'esc' button to exit the program. then it will show you what is in 'brand_queue' on terminal
  """