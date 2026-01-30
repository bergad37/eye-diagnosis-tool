from PIL import Image, ImageDraw
from pathlib import Path

# 1️⃣ Output path for dummy image
output_path = Path("data/dummy_eye.jpg")
output_path.parent.mkdir(exist_ok=True)  # create folder if it doesn't exist

# 2️⃣ Create a blank RGB image (white background)
img = Image.new("RGB", (224, 224), color=(255, 255, 255))

# 3️⃣ Draw a simple eye-like circle
draw = ImageDraw.Draw(img)
center = (112, 112)  # center of the image
iris_radius = 50
pupil_radius = 20

# Draw iris (blue-ish)
draw.ellipse(
    [center[0]-iris_radius, center[1]-iris_radius,
     center[0]+iris_radius, center[1]+iris_radius],
    fill=(30, 144, 255)
)

# Draw pupil (black)
draw.ellipse(
    [center[0]-pupil_radius, center[1]-pupil_radius,
     center[0]+pupil_radius, center[1]+pupil_radius],
    fill=(0, 0, 0)
)

# 4️⃣ Save the image
img.save(output_path)
print(f"✅ Dummy eye image saved to {output_path}")
