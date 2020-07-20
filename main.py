import boto3
import cv2
import json
import numpy as np
import os
from aws_utilities import TextrackUtilities


def do_find_all_documents(path, extension):
    documents = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if extension in file:
                # print(os.path.join(r, file))
                documents.append(os.path.join(r, file))
    return documents


if __name__ == '__main__':
    utils = TextrackUtilities(boto3=boto3, cv2=cv2)
    path = "./Photos/"
    extension = '.jpg'
    documents = do_find_all_documents(path=path, extension=extension)
    # documents = ['./Photos/comprovante.jpg']
    for document in documents:
        print(document)
        image = cv2.imread(document)

        try:
            dimensions = image.shape
            img_size_x = dimensions[0]
            img_size_y = dimensions[1]
            # if img_size_x > 800 and img_size_y > 800:
            if img_size_x > 800 and img_size_y > 800:
                # percent by which the image is resized
                # scale_percent = 25
                scale_percent = 50
                if img_size_x > 800 and img_size_y > 800:
                    scale_percent = 25
                # calculate the scale_percent percent of original dimensions
                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)
                # resize image
                image = cv2.resize(image, (width, height))

            try:
                with open(f"{document[:-4]}.txt", 'rb') as doc:
                    resp = json.load(doc)
            except FileNotFoundError:
                print("Consultando AWS")
                resp = utils.do_textTract(document)

            if resp:
                all_text = utils.get_all_text(resp)
                print(all_text)
                img = utils.do_bonding_boxes(resp, image, 2)
                img = utils.do_bonding_boxes_numeration(
                    img=img,
                    response=resp,
                    character_threshold=3
                )
                cv2.imshow(document, img)
                cv2.waitKey()
                cv2.destroyWindow(document)

            else:
                print("No RESP :(")
        except AttributeError:
            print("Parece que a imagem nao pode ser carregada")



