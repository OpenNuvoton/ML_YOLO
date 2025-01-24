"""
json format to txt annotation
"""

import json
import os
import shutil
import re


class Json2Txt:
    """
    Json2Txt class
    """

    def __init__(self, json_file_name, custom_dataset_output_yolo, output_ann_path, category_class_name):
        self.create_folder(custom_dataset_output_yolo)
        self.create_folder(output_ann_path)
        self.data = self.__open_json(json_file_name)
        self.category_class_name = category_class_name
        self.category_class_num = self.__transfer_classes(self.category_class_name)

    def create_folder(self, dir_path):
        """
        create folder
        """
        try:
            os.mkdir(dir_path)
        except OSError as error:
            print(error)
            print("skip create")

    def __open_json(self, json_file_name):
        f = open(json_file_name, encoding="utf-8")
        data = json.load(f)
        f.close()
        return data

    def __load_images_from_folder(self, folder):  ### 使用yield，降低ram使用
        count = 0
        for filename in os.listdir(folder):
            count += 1
            yield filename

    def __transfer_classes(self, category_class_name):  ### 轉換label name to number

        category_class_num = []
        classes = self.data["categories"]

        if category_class_name == ["all"]:
            return category_class_name

        for name in category_class_name:
            for d in classes:
                if name == d["name"]:
                    category_class_num.append(d["id"])
        print(category_class_num, category_class_name)
        return category_class_num

    def get_img(self, filename):
        """
        get image
        """
        for img_dict in self.data["images"]:
            if img_dict["file_name"] == filename:
                return img_dict

    def get_img_ann(self, image_id):
        """
        get image annotation
        """
        img_ann = []
        is_found = False
        for ann in self.data["annotations"]:
            if ann["image_id"] == image_id:
                img_ann.append(ann)
                is_found = True
        if is_found:
            return img_ann
        return None

    def convert_to_yolo(self, ann, img_w, img_h, file_object, coco_fix=True):
        """
        convert to yolo
        """
        original_category = ann["category_id"]
        current_category = ann["category_id"] - 1  # As yolo format labels start from 0

        if (original_category in self.category_class_num) or (self.category_class_num == ["all"]):  # if the obj is in the wanted list

            # Remove the empty categories from 90 to 80
            shift_val = 0
            if coco_fix:
                if current_category > 11:
                    shift_val -= 1
                if current_category > 25:
                    shift_val -= 1
                if current_category > 29:
                    shift_val -= 2
                if current_category > 44:
                    shift_val -= 1
                if current_category > 65:
                    shift_val -= 1
                if current_category > 68:
                    shift_val -= 2
                if current_category > 70:
                    shift_val -= 1
                if current_category > 82:
                    shift_val -= 1
            current_category += shift_val

            current_bbox = ann["bbox"]
            x = current_bbox[0]
            y = current_bbox[1]
            w = current_bbox[2]
            h = current_bbox[3]

            # Finding midpoints
            x_centre = (x + (x + w)) / 2
            y_centre = (y + (y + h)) / 2

            # Normalization
            x_centre = x_centre / img_w
            y_centre = y_centre / img_h
            w = w / img_w
            h = h / img_h

            # Limiting upto fix number of decimal places
            x_centre = format(x_centre, ".6f")
            y_centre = format(y_centre, ".6f")
            w = format(w, ".6f")
            h = format(h, ".6f")

            # Writing current yolo ann txt
            if self.category_class_num == ["all"]:
                file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
            else:  # need to re-organized the index if not all class
                for idx_adj, val in enumerate(self.category_class_num):
                    if original_category == val:
                        break
                file_object.write(f"{idx_adj} {x_centre} {y_centre} {w} {h}\n")

    def create_classes(self, classes_file_name):
        """
        create classes
        """
        file_object = open(f"{classes_file_name}", "a", encoding="utf-8")
        if self.category_class_num == ["all"]:
            classes = self.data["categories"]
            for d in classes:
                file_object.write(f"{d['name']}\n")
        else:
            for name in self.category_class_name:
                file_object.write(f"{name}\n")

        file_object.close()
        print(f"Create the classes file: {classes_file_name}")

    def create_dataset_path_txt(self, new_imgs_folder, output_dataset_path):
        """
        create dataset path txt
        """
        with open(f"{output_dataset_path}", "a", encoding="utf-8") as file_object:
            for img_filename in self.__load_images_from_folder(new_imgs_folder):
                # if re.search('[0-9]*.jpg', img_filename):
                if re.search(r"(?i)([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$", img_filename):
                    file_object.write(f"{os.path.join(new_imgs_folder, img_filename)}\n")

        print(f"Create the dataset path file: {output_dataset_path}")

    def run_annotation(self, imgs_folder, output_path):
        """
        run annotation
        """
        for img_filename in self.__load_images_from_folder(imgs_folder):
            img = self.get_img(img_filename)
            img_ann = self.get_img_ann(img["id"])
            if img_ann:
                # Opening yolo ann file for current image
                file_object = open(f"{output_path}/{img_filename.split('.')[0]}.txt", "a", encoding="utf-8")
                # loop all annotation
                for ann in img_ann:
                    self.convert_to_yolo(ann, img["width"], img["height"], file_object)
                file_object.close()
                # copy img to new output_path because yolo train dir structure needed
                try:
                    shutil.copy(os.path.join(imgs_folder, img_filename), os.path.join(output_path, img_filename))
                except FileNotFoundError:
                    print("Error occurred while copying file.")
            else:
                print(f"Json file doesn't have {img} annotation info")
        print(f"Create the yolo annotation files in: {output_path}")
