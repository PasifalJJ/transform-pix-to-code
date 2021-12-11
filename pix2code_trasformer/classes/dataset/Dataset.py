from __future__ import print_function

__author__ = 'jsc'

import os

from ..Vocabulary import *
from ..Utils import *
from ..model.Config import *

class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_size = None
        # 文件名称数组
        self.ids = []
        # 归一化文件信息数组
        self.input_images = []
        # 区间context(48)长度内的字符
        self.partial_sequences = []
        # context(48)字符后的下一个字符
        self.next_words = []

        self.voc = Vocabulary()
        self.size = 0
        self.enc_inputs = []
        self.dec_inputs = []
        self.dec_outputs = []
        self.dec_out_sequences = []

    @staticmethod
    def load_paths_only(path):
        print("Parsing data...")
        gui_paths = []
        img_paths = []
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                path_gui = "{}/{}".format(path, f)
                gui_paths.append(path_gui)
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    path_img = "{}/{}.png".format(path, file_name)
                    img_paths.append(path_img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    path_img = "{}/{}.npz".format(path, file_name)
                    img_paths.append(path_img)

        assert len(gui_paths) == len(img_paths)
        return gui_paths, img_paths

    def load(self, path, generate_binary_sequences=False):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                gui = open("{}/{}".format(path, f), 'r')
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    img = Utils.get_preprocessed_img("{}/{}.png".format(path, file_name), IMAGE_SIZE)
                    self.append(file_name, gui, img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    img = np.load("{}/{}.npz".format(path, file_name))["features"]
                    self.append(file_name, gui, img)

        print("数据数字化表示...")

        print("生成稀疏向量，数据向量化表示...")
        self.voc.create_binary_representation()  # 将voc中的token进行稀疏矩阵
        if generate_binary_sequences:
            self.partial_sequences = self.binarize(self.partial_sequences, self.voc)  # 将content(48)序列中的token稀疏化
            self.dec_out_sequences = self.binarize(self.dec_out_sequences, self.voc)  # 将content(48)序列中的token稀疏化
            self.next_words = self.sparsify_labels(self.next_words, self.voc)
        else:
            self.partial_sequences = self.indexify(self.partial_sequences, self.voc)  # 将content(48)序列中的token使用数字进行表示
            self.dec_out_sequences = self.indexify(self.dec_out_sequences, self.voc)  # 将content(48)序列中的token使用数字进行表示
            self.next_words = self.indexify_labels(self.next_words, self.voc)

        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))

        self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size

        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

    def convert_arrays(self):
        print("Convert arrays...")
        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self, sample_id, gui, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            Utils.show(pic)
        # 将 224*224*3 转变为 3*224*224
        img = np.transpose(img, (2, 0, 1))

        tem_enc_inputs = [START_TOKEN]
        tem_dec_inputs = [START_TOKEN]
        tem_dec_outputs = []
        token_sequence = [START_TOKEN]
        for line in gui:
            line = line.replace(",", " ").replace("\n", " ")
            line = ' '.join(line.split())
            tokens = line.split(" ")
            for token in tokens:
                self.voc.append(token)
                token_sequence.append(token)
                tem_enc_inputs.append(token)
                tem_dec_inputs.append(token)
                tem_dec_outputs.append(token)
        token_sequence.append(END_TOKEN)
        tem_dec_outputs.append(END_TOKEN)
        self.enc_inputs.append(tem_enc_inputs)
        self.dec_inputs.append(tem_dec_inputs)
        self.dec_outputs.append(tem_dec_outputs)

        for j in range(len(tem_enc_inputs)):
            pl = [PLACEHOLDER] * CONTEXT_LENGTH
            pl[0:j+1] = tem_enc_inputs[0:j+1]
            context = np.array(pl)

            pl_out = [PLACEHOLDER] * CONTEXT_LENGTH
            pl_out[0:j + 1] = tem_dec_outputs[0:j + 1]
            dec_out_context = np.array(pl_out)

            label = tem_dec_outputs[j]
            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.dec_out_sequences.append(dec_out_context)
            self.next_words.append(label)

    @staticmethod
    def indexify(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def binarize(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.binary_vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def sparsify_labels(next_words, voc):
        temp = []
        for label in next_words:
            temp.append(voc.binary_vocabulary[label])

        return temp

    @staticmethod
    def indexify_labels(next_words, voc):
        temp = []
        for label in next_words:
            temp.append(voc.vocabulary[label])

        return temp

    def save_metadata(self, path):
        np.save("{}/meta_dataset".format(path), np.array([self.input_shape, self.output_size, self.size]))
