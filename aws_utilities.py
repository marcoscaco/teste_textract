import json


class TextrackUtilities:
    boto3 = None
    cv2 = None

    def __int__(self, boto3):
        self.boto3 = boto3

    def __init__(self, boto3, cv2):
        self.boto3 = boto3
        self.cv2 = cv2

    def do_textTract(self, documentName):
        # Read document content
        with open(documentName, 'rb') as document:
            imageBytes = bytearray(document.read())

        # Amazon Textract client
        textract = self.boto3.client('textract')

        # Call Amazon Textract
        response = textract.detect_document_text(Document={'Bytes': imageBytes})

        print(response)
        with open(f"{documentName[:-4]}.txt", 'w+') as out:
            json.dump(response, out)

        return response

    def do_form_textrack(self, documentName):
        # Read document content
        with open(documentName, 'rb') as document:
            imageBytes = bytearray(document.read())

        # Amazon Textract client
        textract = self.boto3.client('textract')

        # Call Amazon Textract
        response = textract.detect_document_text (Document={'Bytes': imageBytes})

        # Print text
        print("\nText\n========")
        text = ""
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                print('\033[94m' + item["Text"] + '\033[0m')
                text = text + " " + item["Text"]

        # Amazon Comprehend client
        comprehend = self.boto3.client('comprehend')

        # Detect sentiment
        sentiment = comprehend.detect_sentiment(LanguageCode="en", Text=text)
        print("\nSentiment\n========\n{}".format(sentiment.get('Sentiment')))

        # Detect entities
        entities = comprehend.detect_entities(LanguageCode="en", Text=text)
        print("\nEntities\n========")
        for entity in entities["Entities"]:
            print("{}\t=>\t{}".format(entity["Type"], entity["Text"]))

        return response

    def do_bonding_boxes_numeration(self, response, img, character_threshold=3):
        contador = 0
        dimensions = img.shape
        img_size_x = dimensions[0]
        img_size_y = dimensions[1]
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                if len(item["Text"]) >= character_threshold:
                    text_top_pixel = int(img_size_x * item["Geometry"]["BoundingBox"]["Top"])
                    text_left_pixel = int(img_size_y * item["Geometry"]["BoundingBox"]["Left"])
                    text_height_pixel = int(img_size_x * item["Geometry"]["BoundingBox"]["Height"])
                    text_width_pixel = int(img_size_y * item["Geometry"]["BoundingBox"]["Width"])
                    end_point = (text_left_pixel + text_width_pixel, text_top_pixel + text_height_pixel)
                    font = self.cv2.FONT_HERSHEY_SIMPLEX
                    self.cv2.putText(img, f'{contador}', end_point, font, 1, (0, 255, 0), 2, self.cv2.LINE_AA)
                    contador += 1
        return img

    def do_bonding_boxes(self, response, img, character_threshold=3):
        output = img
        dimensions = img.shape
        img_size_x = dimensions[0]
        img_size_y = dimensions[1]
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                if len(item["Text"]) >= character_threshold:
                    # print(item["Geometry"])
                    text_top_pixel = int(img_size_x * item["Geometry"]["BoundingBox"]["Top"])
                    text_left_pixel = int(img_size_y * item["Geometry"]["BoundingBox"]["Left"])
                    text_height_pixel = int(img_size_x * item["Geometry"]["BoundingBox"]["Height"])
                    text_width_pixel = int(img_size_y * item["Geometry"]["BoundingBox"]["Width"])
                    # Blue color in BGR
                    color = (0, 0, 255)
                    # Line thickness of 2 px
                    thickness = 2
                    start_point = (text_left_pixel, text_top_pixel)
                    end_point = (text_left_pixel + text_width_pixel, text_top_pixel + text_height_pixel)
                    # Using cv2.rectangle() method
                    # Draw a rectangle with blue line borders of thickness of 2 px
                    output = self.cv2.rectangle(output, start_point, end_point, color, thickness)

        return output

    def get_all_text(self, response, character_threshold=3):
        phrases = {}
        contador = 0
        # Print detected text
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                if len(item["Text"]) >= character_threshold:
                    phrases[f'{contador}'] = item["Text"]
                    contador += 1

        return phrases
