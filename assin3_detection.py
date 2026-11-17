#!/usr/bin/env python3
# ELEC7023 - Assignment 3

import os

import jetson.inference
import jetson.utils

# Input files
folder = os.path.dirname(os.path.abspath(__file__))
images = sorted(name for name in os.listdir(folder)
                if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")
                and os.path.isfile(os.path.join(folder, name)))
thresholds = [0.3, 0.7]
output_folder = os.path.join(folder, "results")

print("Input folder:", folder)
print("Images:", images)
if len(images) != 2:
    raise ValueError("Place exactly two JPG, JPEG, PNG or BMP images beside this script.")

os.makedirs(output_folder, exist_ok=True)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.3)
summary = []

with open(os.path.join(output_folder, "detection_results.txt"),
          "w", encoding="utf-8") as report:
    for threshold in thresholds:
        net.SetThreshold(threshold)

        for image_index, image_name in enumerate(images, start=1):
            # Reload the original image for each threshold.
            img = jetson.utils.loadImage(os.path.join(folder, image_name))
            if img is None:
                raise RuntimeError("Could not load image: " + image_name)

            detections = net.Detect(img, overlay="box,labels,conf")

            heading = "\nImage: {} | Threshold: {} | Objects: {}\n".format(
                image_name, threshold, len(detections))
            print(heading)
            report.write(heading)

            for i, detection in enumerate(detections, start=1):
                text = "Object {}: {}\n{}\n".format(
                    i, net.GetClassDesc(detection.ClassID), detection)
                print(text)
                report.write(text)

            output_name = "image{}_threshold_{:.1f}.jpg".format(
                image_index, threshold)
            output_path = os.path.join(output_folder, output_name)
            jetson.utils.cudaDeviceSynchronize()
            jetson.utils.saveImage(output_path, img)
            print("Saved:", output_path)
            summary.append((image_name, threshold, len(detections)))

    report.write("\nSummary\nImage\tThreshold\tNumber of detected objects\n")
    print("\nSummary\nImage\tThreshold\tNumber of detected objects")
    for image_name, threshold, count in sorted(summary):
        line = "{}\t{}\t{}\n".format(image_name, threshold, count)
        report.write(line)
        print(line, end="")

