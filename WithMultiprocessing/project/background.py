import cv2
import numpy as np
import os
import time
from multiprocessing import Pool
from datetime import datetime

def remove_background_from_image(input_output_paths):
    input_image_path, output_image_path = input_output_paths
    process_id = os.getpid()
    
    start = time.time()
    start_str = datetime.fromtimestamp(start).strftime('%H:%M:%S')
    print(f"process with ID {process_id} processing IMAGE {os.path.basename(input_image_path)} started at {start_str}")
    image = cv2.imread(input_image_path)
    mask = np.zeros(image.shape[:2], np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)
    cv2.grabCut(image, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    cv2.imwrite(output_image_path, result)
    end = time.time()
    end_str = datetime.fromtimestamp(end).strftime('%H:%M:%S')
    print(f"process with ID {process_id}  processing IMAGE {os.path.basename(input_image_path)} ended at {end_str}")
    time_taken = end - start
    print(f"IMAGE {os.path.basename(input_image_path)} took {time_taken} seconds")

if __name__ == "__main__":
    input_dir = r"C:\Users\Hardik\OneDrive\Desktop\Parallel-vs-Sequential-Image-Processing-OS\WithMultiprocessing\project\input"
    output_dir = r"C:\Users\Hardik\OneDrive\Desktop\Parallel-vs-Sequential-Image-Processing-OS\WithMultiprocessing\project\output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    output_paths = [os.path.join(output_dir, os.path.basename(filename)) for filename in image_paths]
    pool = Pool()
    input_output_paths = zip(image_paths, output_paths)

    process_start = time.time()
    pool.map(remove_background_from_image, input_output_paths)
    pool.close()
    pool.join()
    process_end = time.time()
    process_time_taken = process_end - process_start
    print(f"Total process took {process_time_taken} seconds")

