import cv2
import os

def create_video_from_images(image_folder, output_video, fps=2):
    # Get a list of all image filenames in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]

    # Prepare a list to hold tuples of (timestamp, filename)
    image_filenames = []
    for img in images:
        # Remove the file extension to extract the timestamp
        name, ext = os.path.splitext(img)
        try:
            # Convert the filename (timestamp) to an integer
            timestamp = float(name.replace('-','.'))
            image_filenames.append((timestamp, img))
        except ValueError:
            print(f"Skipping file '{img}': filename is not a valid integer timestamp.")
            continue

    # Sort the images by the integer timestamp
    image_filenames.sort(key=lambda x: x[0])

    # Extract the sorted filenames
    sorted_images = [img for _, img in image_filenames]

    # Check if there are images
    if len(sorted_images) == 0:
        print("No valid images found in the folder.")
        return

    # Read the first image to get its dimensions
    first_image_path = os.path.join(image_folder, sorted_images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image '{first_image_path}'.")
        return
    height, width, layers = frame.shape

    print(f"First image dimensions: width={width}, height={height}, layers={layers}")

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try 'XVID' codec
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not video.isOpened():
        print("Error: VideoWriter not opened.")
        return

    frame_count = 0

    # Loop over each image and add it to the video
    for image in sorted_images:
        image_path = os.path.join(image_folder, image)
        print(f"Processing image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image '{image_path}'. Skipping.")
            continue
        # Resize frame if necessary
        if (frame.shape[1], frame.shape[0]) != (width, height):
            print(f"Resizing frame '{image}' from {frame.shape[1]}x{frame.shape[0]} to {width}x{height}")
            frame = cv2.resize(frame, (width, height))
        video.write(frame)
        frame_count += 1
        print(f"Added frame '{image}' to video.")

    # Release the video writer
    video.release()
    print(f"Video saved as '{output_video}' with {frame_count} frames.")

# Example usage
if __name__ == "__main__":
    image_folder = 'Pictures/'  # Replace with the path to your image folder
    output_video = 'output_video.avi'  # Change the extension to .avi
    create_video_from_images(image_folder, output_video)
