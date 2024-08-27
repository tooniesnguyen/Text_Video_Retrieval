# from src.data_retrieval.faiss_search import SearchFaiss
from src.abstraction.search_db import SearchDB
# from src.data_encoder.Internvideo2 import In
# import torch
# import os
#
#
# import warnings
# warnings.filterwarnings("ignore")
#
#
#
# if __name__ == "__main__":
#     clip_model =  CLIPModel(device = "cuda")
#
#     search = SearchFaiss(bin_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/bin/faiss_CLIP_cosine.bin",
#                         json_path = "/home/toonies/Learn/Text_Video_Retrieval/data/dicts/json/keyframes_id_search.json",
#                         encoder_model= clip_model)
#
#     text_query = "The man is wearing a red shirt and yellow hat"
#     img_paths = search.search_query(text_query, k =42, rerank=False)
#
#     search.save_result(img_paths, save_path = "./results")

import googletrans
import translate
import faiss
from src.abstraction.encoder_model import EncoderModel
from langdetect import detect
import json
import os
import matplotlib.pyplot as plt
from src.data_encoder.Internvideo2 import InternVideo2Model
import math
import numpy as np



class Translation:
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate.
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):
        """
        It takes a string as input, and returns a string with all the letters in lowercase
        :param text: The text to be processed
        :return: The text is being returned in lowercase.
        """
        return text.lower()

    def __call__(self, text):
        """
        The function takes in a text and preprocesses it before translation
        :param text: The text to be translated
        :return: The translated text.
        """
        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text


class MyFaiss:
  def __init__(self, bin_file: str, dict_json: str,device = "cpu", mode = "internvideo2"):
    self.index = self.load_bin_file(bin_file)
    self.translate = Translation()
    self.dict_json = self._read_json(dict_json)
    self.device = device
    if mode == "internvideo2":
      self.Internvideo2_model = InternVideo2Model(device="cuda")
  def load_bin_file(self, bin_file: str):
    return faiss.read_index(bin_file)

  def _read_json(self, file_json):
    with open(file_json, "r") as file:
      data = json.load(file)
    return data

  def show_images(self, image_paths):
    fig = plt.figure(figsize=(15, 10))
    columns = int(math.sqrt(len(image_paths)))
    rows = int(np.ceil(len(image_paths)/columns))

    for i in range(1, columns*rows +1):
      img = plt.imread(image_paths[i - 1])
      ax = fig.add_subplot(rows, columns, i)
      ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

      plt.imshow(img)
      plt.axis("off")

    plt.show()

  def text_search(self, text, k):
    if detect(text) == 'vi':
      text = self.translate(text)
    print("Text translation: ", text)
    text_features = self.Internvideo2_model.text_encoder(text)
    scores, idx_image = self.index.search(text_features, k=k)
    print("Idx images", idx_image)
    result_strings = list(map(lambda idx: self.dict_json[idx] if 0 <= idx < len(self.dict_json) else None, idx_image[-1]))
    return result_strings


def main():

  ##### TESTING #####
  bin_file=f"/home/nhi/HCMAI/Text_Video_Retrieval/dicts/faiss_InternVideo2_cosine.bin"
  json_path = f"/home/nhi/HCMAI/Text_Video_Retrieval/dicts/keyframes_id_search_internVid2.json"

  cosine_faiss = MyFaiss(bin_file, json_path)


  ##### TEXT SEARCH #####
  text = ('Lũ lụt nhấn chìm rất nhiều cây và nhà cửa')
  # text = 'Hình ảnh của Nhân mập'


  image_paths = cosine_faiss.text_search(text, k=24)
  # cosine_faiss.write_csv(infos_query, des_path_submit='./')
  base_path = f"/home/nhi/HCMAI/Text_Video_Retrieval/data/"

  # Tạo đường dẫn tuyệt đối cho từng hình ảnh
  img_paths = [os.path.join(base_path, image_path) for image_path in image_paths]
  cosine_faiss.show_images(img_paths)

if __name__ == "__main__":
    main()