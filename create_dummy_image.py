from PIL import Image, ImageDraw

def create_dummy_image():
    img = Image.new('RGB', (100, 100), color = 'red')
    d = ImageDraw.Draw(img)
    d.text((10,10), "Hello World", fill=(255,255,0))
    img.save('d:/image_python/images/dummy.jpg')
    print("Created dummy.jpg")

if __name__ == "__main__":
    create_dummy_image()
