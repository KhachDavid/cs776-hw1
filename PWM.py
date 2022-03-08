import numpy as np
from util import *

class PWMList(object):
    def __init__(self, motif_list, motif_width, alphabet_list, pi, pi_not_matching):
        self.alphabet_list = alphabet_list
        self.motif_list = motif_list
        self.motif_width = motif_width
        self.pi = pi
        self.pi_not_matching = pi_not_matching
        
        self.pwm = self.create_pwm()

    def __len__(self):
        return len(self.motif_list)

    def __str__(self):
        # loop over each key in the dictionary
        rtrn = ""
        for key in self.pwm:
            # loop over each row in the 2d numpy array
            rtrn += "\nPWM for "+ key + ":\n\n"
            for index, row in enumerate(self.pwm[key]):
                # loop over each element in the row
                rtrn += self.alphabet_list[index] + " | "
                for element in row:
                    # round element to 3 decimal places
                    rtrn += str(round(element, 3)) + " "
                rtrn += "\n"
            rtrn += "\n"
        return rtrn

    def __repr__(self):
        return str(self.pwm)

    def __eq__(self, other):
        return self.pwm == other.pwm

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

    def keys(self):
        return self.pwm.keys()

    def get_pwm(self):
        return self.pwm

    def get_motif_list(self):
        return self.motif_list

    def get_motif_width(self):
        return self.motif_width

    def get_alphabet_list(self):
        return self.alphabet_list

    def get_pi(self):
        return self.pi

    def get_pi_not_matching(self):
        return self.pi_not_matching

    def update_pwm(self, motif, base, column, value):
        self.get_pwm()[motif][self.alphabet_list.index(base)][column] = value

    def create_pwm(self):
        pwm = {}
        for motif in self.motif_list:
            pwm[motif] = np.zeros((len(self.alphabet_list), self.motif_width + 1))
            for character in self.alphabet_list:
                pwm[motif][self.alphabet_list.index(character)] = 1 / len(self.alphabet_list)
                for i in range(self.motif_width):
                    if motif[i] == character:
                        pwm[motif][self.alphabet_list.index(character)][i + 1] = self.pi
                    else:
                        pwm[motif][self.alphabet_list.index(character)][i + 1] = self.pi_not_matching
        return pwm
