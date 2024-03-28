images = os.listdir()
IMAGE_SIZE = (256, 256)

for image_name in images:
    image = cv2.imread(image_name)
    
    resized_image = cv2.resize(image, IMAGE_SIZE)
    
    os.chdir("..")
    os.chdir("resizedImage")
    
    new_name = f"{str(uuid.uuid4())}.jpg"
    cv2.imwrite(new_name, resized_image)
    
    os.chdir("..")
    os.chdir("fetusImage")